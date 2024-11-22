import pandas as pd
import numpy as np
import plotly.express as px

# Generate random dates between 2018 and 2023
dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='M')

# Generate random computer sales data
sales = np.random.randint(100, 1000, size=len(dates))

# Create a DataFrame
df = pd.DataFrame({'Date': dates, 'Sales': sales})

# Create a time series plot using Plotly
fig = px.line(df, x='Date', y='Sales', title='Computer Sales Time Series (2018-2023)')

# Add labels to the x and y axes
fig.update_layout(xaxis_title='Date', yaxis_title='Sales')

# Add gridlines
fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)

# Add legend
fig.update_layout(showlegend=True)

# Add trendline
fig.add_trace(px.line(df, x='Date', y=df['Sales'].rolling(window=12).mean(), name='12-month Moving Average').data[0])

# Add annotations
fig.add_annotation(x='2021-01-01', y=800, text='Peak Sales', showarrow=True, arrowhead=1)

fig.show()