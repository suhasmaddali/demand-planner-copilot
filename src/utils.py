import pandas as pd
from datetime import datetime
import streamlit as st

def adjust_crd_date(filename: str, dataframe):
    try:
        # Assuming filename starts with 'CustAllocDownload' and the date is the next 8 characters
        date_str_index = filename.index("CustAllocDownload") + len("CustAllocDownload")
        file_date_str = filename[date_str_index:date_str_index + 8]
        
        # Parse the extracted date and format it to 'YYYY-MM-DD'
        file_date = datetime.strptime(file_date_str, '%Y%m%d')
        file_date_formatted = file_date.strftime('%Y-%m-%d')
    except (ValueError, IndexError):
        # Handle cases where the filename does not contain a valid date
        raise ValueError(f"Filename '{filename}' does not contain a valid date in the expected format 'CustAllocDownloadYYYYMMDD'.")

    # Convert 'CRD Date' column to datetime format for comparison, handling errors
    dataframe['CRD Date'] = pd.to_datetime(dataframe['CRD Date'], format='%Y-%m-%d', errors='coerce')

    # Adjust 'CRD Date' based on the opposite comparison with file date
    dataframe['CRD Date Adjusted'] = dataframe['CRD Date'].apply(
        lambda x: file_date_formatted if pd.notnull(x) and x < file_date else (x.strftime('%Y-%m-%d') if pd.notnull(x) else pd.NaT)
    )

    return dataframe


def map_calendar_to_fiscal(data_time_profile, data):
    data_time_profile['Day'] = pd.to_datetime(data_time_profile['Day'], format='%m/%d/%Y')
    data_time_profile['Weekly Dates'] = data_time_profile['Day'].apply(
        lambda x: x + pd.Timedelta(days=(6 - x.weekday()))
    )
    data_time_profile['Weekly Dates'] = data_time_profile['Weekly Dates'].apply(
        lambda x: f"W/e {x.strftime('%d %b %y')}"
    )
    fiscal_mapping = data_time_profile.set_index('Day')[['Fiscal Month', 'Fiscal Quarter', 'Fiscal Year', 'Weekly Dates']]
    date_columns = ['Ord Date', 'CRD Date', 'CRD Date Adjusted', 'Inv Date', 'Pl. GI Date', 'MAD', 'Allocation Date']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], format='%Y-%m-%d')

    for col in date_columns:
        data[f'{col} Fiscal Month'] = data[col].map(fiscal_mapping['Fiscal Month'])
        data[f'{col} Fiscal Quarter'] = data[col].map(fiscal_mapping['Fiscal Quarter'])
        data[f'{col} Fiscal Year'] = data[col].map(fiscal_mapping['Fiscal Year'])
        data[f'{col} Weekly Dates'] = data[col].map(fiscal_mapping['Weekly Dates'])

    return data

@st.fragment
def index_time_profile(df_time_profile, current_fiscal_quarter):

    unique_values = list(df_time_profile[df_time_profile['Fiscal Quarter'] == current_fiscal_quarter]['Weekly Dates'].unique()) + [current_fiscal_quarter]
    index_primary = ['BUF', 'JFF', 'Demand', 'Shipped', 'Supply', 'Allocation']
    index = pd.MultiIndex.from_product([index_primary, ['Total']])
    df_result = pd.DataFrame(0, index=index, columns=unique_values)

    return df_result

@st.fragment
def convert_to_anaplan(df_date_converted, df_result, current_fiscal_quarter):
    
    for demand_type in df_date_converted['Demand Type'].unique():
        if demand_type == 'Allocation':
            df_demand_type = df_date_converted[df_date_converted['Demand Type'] == demand_type].copy()
            df_fiscal_date = df_demand_type[df_demand_type['Allocation Date Fiscal Quarter'] == current_fiscal_quarter].copy()
            grouped_sum = df_fiscal_date.groupby(by=['Allocation Date Weekly Dates'])['Allocation Qty'].sum()
            for weekly_date, qty in grouped_sum.items():
                if weekly_date in df_result.columns:
                    df_result.loc[('Allocation', 'Total'), weekly_date] = qty

        elif demand_type == 'BUF':
            df_demand_type = df_date_converted[df_date_converted['Demand Type'] == demand_type].copy()
            df_fiscal_date = df_demand_type[df_demand_type['CRD Date Fiscal Quarter'] == current_fiscal_quarter].copy()
            grouped_sum = df_fiscal_date.groupby(by=['CRD Date Weekly Dates'])['Quantity'].sum()
            for weekly_date, qty in grouped_sum.items():
                if weekly_date in df_result.columns:
                    df_result.loc[('BUF', 'Total'), weekly_date] = qty

        elif demand_type == 'JFF':
            df_demand_type = df_date_converted[df_date_converted['Demand Type'] == demand_type].copy()
            df_fiscal_date = df_demand_type[df_demand_type['CRD Date Fiscal Quarter'] == current_fiscal_quarter].copy()
            grouped_sum = df_fiscal_date.groupby(by=['CRD Date Weekly Dates'])['Quantity'].sum()
            for weekly_date, qty in grouped_sum.items():
                if weekly_date in df_result.columns:
                    df_result.loc[('JFF', 'Total'), weekly_date] = qty

        elif demand_type == 'Supply':
            df_demand_type = df_date_converted[df_date_converted['Demand Type'] == demand_type].copy()
            df_fiscal_date = df_demand_type[df_demand_type['CRD Date Fiscal Quarter'] == current_fiscal_quarter].copy()
            grouped_sum = df_fiscal_date.groupby(by=['CRD Date Weekly Dates'])['Quantity'].sum()
            for weekly_date, qty in grouped_sum.items():
                if weekly_date in df_result.columns:
                    df_result.loc[('Supply', 'Total'), weekly_date] = qty

        elif demand_type == 'SHIPPED':
            df_demand_type = df_date_converted[df_date_converted['Demand Type'] == demand_type].copy()
            df_fiscal_date = df_demand_type[df_demand_type['Pl. GI Date Fiscal Quarter'] == current_fiscal_quarter].copy()
            grouped_sum = df_fiscal_date.groupby(by=['Pl. GI Date Weekly Dates'])['Quantity'].sum()
            for weekly_date, qty in grouped_sum.items():
                if weekly_date in df_result.columns:
                    df_result.loc[('Shipped', 'Total'), weekly_date] = qty

        elif demand_type == 'BOOKED':
            df_demand_type = df_date_converted[df_date_converted['Demand Type'] == demand_type].copy()
            df_fiscal_date = df_demand_type[df_demand_type['CRD Date Adjusted Fiscal Quarter'] == current_fiscal_quarter].copy()
            grouped_sum = df_fiscal_date.groupby(by=['CRD Date Adjusted Weekly Dates'])['Quantity'].sum()
            for weekly_date, qty in grouped_sum.items():
                if weekly_date in df_result.columns:
                    df_result.loc[('Demand', 'Total'), weekly_date] = qty

    for weekly_date in df_result.columns:
        if all([weekly_date in df_result.columns]):
            df_result.loc[('Allocation - Demand Delta', 'Total'), weekly_date] = (
                df_result.loc[('Allocation', 'Total'), weekly_date]
                - df_result.loc[('Demand', 'Total'), weekly_date]
            )
            df_result.loc[('Supply - Demand Delta', 'Total'), weekly_date] = (
                df_result.loc[('Supply', 'Total'), weekly_date]
                - df_result.loc[('Demand', 'Total'), weekly_date]
            )

            df_result.loc[('Supply - Allocation Delta', 'Total'), weekly_date] = (
                df_result.loc[('Supply', 'Total'), weekly_date]
                - df_result.loc[('Allocation', 'Total'), weekly_date]
            )

    try:
        final_column = df_result.columns[-1]
        df_result.drop([final_column], axis=1, inplace=True)
        final_column_list = final_column.split('-')
        renamed_column = final_column_list[1] + " " + final_column_list[0]
        df_result[renamed_column] = df_result.sum(axis=1)
    except:
        final_column = df_result.columns[-1]
        df_result.drop([final_column], axis=1, inplace=True)
        final_column_list = final_column.split('-')[0]
        renamed_column = final_column_list[1] + " " + final_column_list[0]
        df_result[renamed_column] = df_result.sum(axis=1)


    output_df = df_result.copy()

    return output_df

