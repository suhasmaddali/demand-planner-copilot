# import os
# import urllib.parse
# import logging
# import httpx
# import streamlit as st
# from dotenv import load_dotenv
# from fastapi import HTTPException
# import os
# import urllib.parse
# import httpx
# import json
# import requests
# import logging
# import time

# load_dotenv()

# logger = logging.getLogger(__name__)

# # Initialize local storage
# log = logging.getLogger()
# MAX_RETRY_COUNT = 3
# SLEEP_TIME = 2

# class AzureADOAuthProvider:
#     """
#     Manages authentication by using Azure active directory. This class is intended to be used
#     for the following authentication workflow. For more details on how this is performed, see
#     `check_login` in `components.py`.
#     """

#     def __init__(self):
#         self.client_id = os.environ.get("OAUTH_AZURE_AD_CLIENT_ID")
#         self.client_secret = os.environ.get("OAUTH_AZURE_AD_CLIENT_SECRET")
    
#         if os.getenv("AZURE_EXTENSION_DIR") is None:
#             logging.info("Loading AAD redirect URI for local deployment")
#             self.redirect_uri = os.environ.get("OAUTH_AZURE_AD_REDIRECT_URI_LOCAL")
#         else :
#             logging.info("Loading AAD redirect URI for container deployment")
#             self.redirect_uri = os.environ.get("OAUTH_AZURE_AD_REDIRECT_URI_APP_SERVICE")
            
#         self.tenant_id = os.environ.get("OAUTH_AZURE_AD_TENANT_ID")
#         self.authorize_url = (
#             f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/authorize"
#         )
#         self.token_url = (
#             f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
#         )
#         self.scope = "https://graph.microsoft.com/User.Read"
#         self.response_mode = "query"

#     def get_authorize_url(self):
#         params = {
#             "client_id": self.client_id,
#             "response_type": "code",
#             "redirect_uri": self.redirect_uri,
#             "response_mode": self.response_mode,
#             "scope": self.scope,
#         }
#         return f"{self.authorize_url}?{urllib.parse.urlencode(params)}"

#     @st.cache_resource(show_spinner=False)
#     def get_token(_self, code: str):
#         payload = {
#             "client_id": _self.client_id,
#             "client_secret": _self.client_secret,
#             "code": code,
#             "grant_type": "authorization_code",
#             "redirect_uri": _self.redirect_uri,
#         }

#         with httpx.Client() as client:
#             response = client.post(_self.token_url, data=payload)
#             try:
#                 response.raise_for_status()
#             except httpx.HTTPStatusError as exc:
#                 raise HTTPException(
#                     status_code=400, detail=f"Error fetching token: {exc.response.text}"
#                 )
#             json_response = response.json()
#             token = json_response.get("access_token")
#             if not token:
#                 raise HTTPException(
#                     status_code=400, detail="Failed to get the access token"
#                 )
#             return token

#     @st.cache_resource(show_spinner=False)
#     def get_user_info(_self, token: str):
#         with httpx.Client() as client:
#             response = client.get(
#                 "https://graph.microsoft.com/v1.0/me",
#                 headers={"Authorization": f"Bearer {token}"},
#             )
#             response.raise_for_status()
#             return response.json()

# def handle_connection_error(retry_count: int, url: str, error: str) -> int:
#     if retry_count < 3:
#         retry_count += 1
#         log.warning(
#             "Got connection error: %s. Retrying in %s seconds", error, SLEEP_TIME
#         )
#         time.sleep(SLEEP_TIME)
#     else:
#         log.error(
#             "Exceeded max retries. Error getting response from api %s: %s", url, error
#         )

#     return retry_count


# def get_data(url: str, headers: dict, params=None, timeout=None, body=None) -> dict:
#     retry_count = 0
#     while retry_count < MAX_RETRY_COUNT:
#         try:
#             api_response = requests.get(
#                 url,
#                 headers=headers,
#                 params=params,
#                 timeout=timeout,
#                 data=json.dumps(body),
#             )
#             if api_response.ok:
#                 return api_response.json()
#             elif str(api_response.status_code).startswith("5"):
#                 retry_count = handle_connection_error(
#                     retry_count, url, api_response.text
#                 )
#             else:
#                 raise Exception(api_response.text)
#         except ConnectionError as ex:
#             retry_count = handle_connection_error(retry_count, url, str(ex))
#         except Exception as ex:
#             log.error("Error getting response from api %s: %s", url, ex)
#             break


# def check_user_eligibility(user_info, dls_list):
#     """Check if user have access to the DL group"""
#     base_url = "https://helios-api.nvidia.com/api"
#     check_user_groups = "v3/groups?filter[descendantUserLogin]={}"
#     token = os.getenv("HELIOS_AUTH_TOKEN")
#     headers = {"auth-token": token}
#     username = user_info.get("mail").split("@nvidia.com")[0]
#     check_user_group_url = f"{base_url}/{check_user_groups.format(username)}"
#     for dl in dls_list:
#         check_user_group_url = f"{check_user_group_url}&filter[names]={dl}"
#     response = get_data(url=check_user_group_url, headers=headers)
#     data = response.get("data") if response else None
#     if data:
#         return True
#     return False




import os
import urllib.parse
import logging
import httpx
import streamlit as st
from dotenv import load_dotenv
from fastapi import HTTPException
import json
import requests
import time

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_RETRY_COUNT = 3
SLEEP_TIME = 2

class AzureADOAuthProvider:
    """
    Manages authentication by using Azure Active Directory. This class is intended to be used
    for the following authentication workflow. For more details on how this is performed, see
    `check_login` in `components.py`.
    """

    def __init__(self):
        self.client_id = os.getenv("OAUTH_AZURE_AD_CLIENT_ID")
        self.client_secret = os.getenv("OAUTH_AZURE_AD_CLIENT_SECRET")
        self.tenant_id = os.getenv("OAUTH_AZURE_AD_TENANT_ID")
        self.scope = "https://graph.microsoft.com/User.Read"
        self.response_mode = "query"

        # Check and set redirect URI based on deployment environment
        if os.getenv("AZURE_EXTENSION_DIR") is None:
            logger.info("Loading AAD redirect URI for local deployment")
            self.redirect_uri = os.getenv("OAUTH_AZURE_AD_REDIRECT_URI_LOCAL")
        else:
            logger.info("Loading AAD redirect URI for container deployment")
            self.redirect_uri = os.getenv("OAUTH_AZURE_AD_REDIRECT_URI_APP_SERVICE")

        # Authorization and token URLs
        self.authorize_url = (
            f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/authorize"
        )
        self.token_url = (
            f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        )


    def get_authorize_url(self):
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "response_mode": self.response_mode,
            "scope": self.scope,
        }
        return f"{self.authorize_url}?{urllib.parse.urlencode(params)}"

    @st.cache_resource(show_spinner=False)
    def get_token(_self, code: str):  # Renamed `self` to `_self`
        payload = {
            "client_id": _self.client_id,
            "client_secret": _self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": _self.redirect_uri,
        }

        with httpx.Client() as client:
            response = client.post(_self.token_url, data=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.error("Error fetching token: %s", exc.response.text)
                raise HTTPException(
                    status_code=400, detail=f"Error fetching token: {exc.response.text}"
                )

            json_response = response.json()
            token = json_response.get("access_token")
            if not token:
                logger.error("Failed to obtain access token")
                raise HTTPException(status_code=400, detail="Failed to get the access token")

            logger.info("Access token successfully obtained")
            return token

    @st.cache_resource(show_spinner=False)
    def get_user_info(_self, token: str):  # Renamed `self` to `_self`
        with httpx.Client() as client:
            response = client.get(
                "https://graph.microsoft.com/v1.0/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.error("Error fetching user info: %s", exc.response.text)
                raise HTTPException(
                    status_code=401, detail=f"Unauthorized access: {exc.response.text}"
                )
            return response.json()

def handle_connection_error(retry_count: int, url: str, error: str) -> int:
    if retry_count < MAX_RETRY_COUNT:
        retry_count += 1
        logger.warning("Connection error: %s. Retrying in %s seconds", error, SLEEP_TIME)
        time.sleep(SLEEP_TIME)
    else:
        logger.error("Exceeded max retries for API %s: %s", url, error)

    return retry_count

def get_data(url: str, headers: dict, params=None, timeout=None, body=None) -> dict:
    retry_count = 0
    while retry_count < MAX_RETRY_COUNT:
        try:
            api_response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=timeout,
                data=json.dumps(body),
            )
            if api_response.ok:
                return api_response.json()
            elif str(api_response.status_code).startswith("5"):
                retry_count = handle_connection_error(retry_count, url, api_response.text)
            else:
                raise Exception(api_response.text)
        except ConnectionError as ex:
            retry_count = handle_connection_error(retry_count, url, str(ex))
        except Exception as ex:
            logger.error("Error fetching API response from %s: %s", url, ex)
            break

def check_user_eligibility(user_info, dls_list):
    """Check if user has access to the DL group"""
    base_url = "https://helios-api.nvidia.com/api"
    check_user_groups = "v3/groups?filter[descendantUserLogin]={}"
    token = os.getenv("HELIOS_AUTH_TOKEN")
    headers = {"auth-token": token}
    username = user_info.get("mail").split("@nvidia.com")[0]
    check_user_group_url = f"{base_url}/{check_user_groups.format(username)}"
    for dl in dls_list:
        check_user_group_url = f"{check_user_group_url}&filter[names]={dl}"
    response = get_data(url=check_user_group_url, headers=headers)
    data = response.get("data") if response else None
    return bool(data)

