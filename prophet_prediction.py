import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.tools as tls
import os
import pickle

raw_df = pd.read_csv(r"c:\Users\avrahamma\Documents\School\AI_for_social_good\paris_bread_sales - Sheet2.csv")

# Prepare the data for Prophet
df = raw_df[["date", "fixed_sold_bread"]].copy()  # Use .copy() to avoid SettingWithCopyWarning
df.columns = ["ds", "y"]  # Rename the columns to match Prophet's requirements

# Convert the date to a datetime object
df.loc[:, "ds"] = pd.to_datetime(df["ds"], format="%d/%m/%Y")

# Convert the sales to a numeric object
df.loc[:, "y"] = pd.to_numeric(df["y"])

model_path = r"c:\Users\avrahamma\Documents\School\AI_for_social_good\prophet_model.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    # Train the model
    model = Prophet()
    model.fit(df)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

# Make a prediction
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)

# Convert the Matplotlib figure to a Plotly figure
plotly_fig = tls.mpl_to_plotly(fig)

# Display the Plotly figure
plot(plotly_fig)

print(forecast.head())
print(forecast.tail())

# Save the forecast to a CSV file
# forecast_path = r"c:\Users\avrahamma\Documents\School\AI_for_social_good\prophet_forecast.csv"
# forecast.to_csv(forecast_path, index=False)
