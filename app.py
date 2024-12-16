import streamlit as st
import pandas as pd
from prophet import Prophet
import datetime
import plotly.graph_objects as go
import pickle
import os
import requests
from typing import Optional, Tuple, Dict


class ModelLoader:
    @staticmethod
    @st.cache_resource
    def load_model(model_path: str = 'model/prophet_model.pkl') -> Optional[Prophet]:
        """Load the trained Prophet model with proper error handling"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file not found at {model_path}")
            return None
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
        
class WeatherService:
    def __init__(self, api_key: str):
        """Initialize weather service with API key"""
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/forecast"
    
    def get_weather_forecasts(self, lat: float, lon: float, dates: list) -> list:
        """
        Get weather forecasts for multiple dates and location
        Returns list of dictionaries with temperature (°C) and precipitation (mm)
        """
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            forecast_data = response.json()
            
            forecasts = []
            for target_date in dates:
                target_datetime = datetime.datetime.combine(target_date, datetime.time(12, 0))
                
                # Get all forecasts for the target date
                day_forecasts = [
                    x for x in forecast_data['list']
                    if datetime.datetime.fromtimestamp(x['dt']).date() == target_date
                ]
                
                if day_forecasts:
                    # Calculate daily mean temperature
                    mean_temp = sum(x['main']['temp'] for x in day_forecasts) / len(day_forecasts)
                    # Sum precipitation for the day
                    total_precip = sum(x.get('rain', {}).get('3h', 0) for x in day_forecasts)
                    
                    forecasts.append({
                        'date': target_date,
                        'temperature': mean_temp,
                        'precipitation': total_precip
                    })
                
            return forecasts
            
        except Exception as e:
            st.error(f"Error fetching weather data: {str(e)}")
            return None

class PredictionManager:
    @staticmethod
    def create_future_df(forecasts: list) -> pd.DataFrame:
        """Create a DataFrame for predictions using weather forecast data"""
        future = pd.DataFrame([{
            'ds': pd.to_datetime(f['date']),
            'temperature_2m_mean': f['temperature'],
            'precipitation_sum_mm': f['precipitation']
        } for f in forecasts])
        
        # Add day dummies
        day_columns = ['friday', 'monday', 'saturday', 'sunday', 'thursday', 'tuesday', 'wednesday']
        for day in day_columns:
            future[f'day_{day}'] = future['ds'].dt.strftime('%A').str.lower() == day
        
        return future

    @staticmethod
    def get_prediction_metrics(forecast: pd.DataFrame) -> dict:
        """Extract and format prediction metrics from the forecast"""
        metrics = []
        for _, row in forecast.iterrows():
            metrics.append({
                'date': row['ds'].date(),
                'day': row['ds'].strftime('%A'),
                'prediction': int(row['yhat']),
                'lower_bound': int(row['yhat_lower']),
                'upper_bound': int(row['yhat_upper']),
                'confidence_range': int((row['yhat_upper'] - row['yhat_lower'])/2)
            })
        return metrics

def main():
    st.set_page_config(
        page_title="3-Day Bread Sales Predictor",
        page_icon="🥖",
        layout="wide"
    )
    
    # Initialize services
    weather_service = WeatherService(st.secrets["OPENWEATHER_API_KEY"])
    
    st.title("🥖 3-Day Bread Sales Predictor")
    
    # Bakery location (could be made configurable)
    PARIS_LAT = 48.8566
    PARIS_LON = 2.3522
    
    # Get next 3 days
    today = datetime.date.today()
    forecast_dates = [today + datetime.timedelta(days=i) for i in range(1, 4)]
    
    # Fetch weather forecasts
    weather_forecasts = weather_service.get_weather_forecasts(PARIS_LAT, PARIS_LON, forecast_dates)
    
    if weather_forecasts:
        # Load model and make predictions
        model = ModelLoader.load_model()
        if model is not None:
            try:
                future = PredictionManager.create_future_df(weather_forecasts)
                forecast = model.predict(future)
                metrics = PredictionManager.get_prediction_metrics(forecast)
                
                # Display results in a grid
                st.header("3-Day Forecast")
                
                cols = st.columns(3)
                for i, (metric, weather, col) in enumerate(zip(metrics, weather_forecasts, cols)):
                    with col:
                        st.subheader(f"Day {i+1}: {metric['day']}")
                        st.metric("Date", metric['date'].strftime('%Y-%m-%d'))
                        st.metric("Temperature", f"{weather['temperature']:.1f}°C")
                        st.metric("Precipitation", f"{weather['precipitation']:.1f}mm")
                        st.metric("Predicted Sales", f"{metric['prediction']} loaves")
                        st.metric("Confidence Range", f"±{metric['confidence_range']} loaves")
                
                # Create visualization
                fig = go.Figure()
                
                # Add prediction points
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='markers+lines',
                    name='Prediction',
                    marker=dict(size=12, color='blue')
                ))
                
                # Add confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(0,0,255,0.2)'),
                    name='Upper Bound'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='rgba(0,0,255,0.2)'),
                    name='Lower Bound'
                ))
                
                fig.update_layout(
                    title='3-Day Bread Sales Forecast',
                    xaxis_title='Date',
                    yaxis_title='Number of Loaves',
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Production recommendations
                st.header("Production Recommendations")
                for metric in metrics:
                    buffer = round(metric['confidence_range'] * 0.2)
                    st.success(f"""
                    📊 **{metric['day']} ({metric['date']}) Production Plan:**
                    - Base prediction: {metric['prediction']} loaves
                    - Recommended buffer: +{buffer} loaves
                    - Total recommended production: {metric['prediction'] + buffer} loaves
                    """)
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")

if __name__ == "__main__":
    main()

