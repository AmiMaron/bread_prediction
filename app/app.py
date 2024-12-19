import streamlit as st
import pandas as pd
from prophet import Prophet
import datetime
import plotly.graph_objects as go
import pickle
import os
import requests
from typing import Optional, Tuple, Dict
from data.date_to_score import get_score_for_date


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
                    if datetime.datetime.utcfromtimestamp(x['dt']).date() == target_date
                ]
                
                if day_forecasts:
                    # Calculate daily mean temperature
                    mean_temp = sum(x['main']['temp'] for x in day_forecasts) / len(day_forecasts)
                    # Sum precipitation for the day
                    total_precip = sum(x.get('rain', {}).get('3h', 0) for x in day_forecasts)
                    
                    # Get holiday score dictionary and extract the score value
                    holiday_info = get_score_for_date(target_date.strftime('%Y-%m-%d'))
                    holiday_score = holiday_info['score']  # Extract just the score value
                    holiday_label = holiday_info['holiday_label']
                    
                    forecasts.append({
                        'date': target_date.strftime('%Y-%m-%d'),
                        'temperature': mean_temp,
                        'precipitation': total_precip,
                        'holiday_score': holiday_score,  # Use the extracted score value
                        'holiday_label': holiday_label
                    })
                
            return forecasts
            
        except Exception as e:
            st.error(f"Error fetching weather data: {str(e)}")
            return None

class PredictionManager:
    @staticmethod
    def create_future_df(forecasts: list) -> pd.DataFrame:
        """Create a DataFrame for predictions using weather forecast and holidays data"""
        future = pd.DataFrame([{
            'ds': pd.to_datetime(f['date']),
            'temperature_2m_mean': f['temperature'],
            'precipitation_sum_mm': f['precipitation'],
            'holiday_score': f['holiday_score'],
            'holiday_label': f['holiday_label']
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
        page_title="Weekly Bread Sales Predictor",
        page_icon="🥖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize services
    weather_service = WeatherService(st.secrets["OPENWEATHER_API_KEY"])
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .holiday-badge {
            background-color: #ff4b4b;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 15px;
            font-size: 0.8rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Weekly Bread Sales Predictor")
    
    
    # Bakery location
    PARIS_LAT = 48.8566
    PARIS_LON = 2.3522
    
    # Get next 7 days
    today = datetime.date.today()
    forecast_dates = [today + datetime.timedelta(days=i) for i in range(1, 8)]
    
    # Fetch weather forecasts
    weather_forecasts = weather_service.get_weather_forecasts(PARIS_LAT, PARIS_LON, forecast_dates)
    
    if weather_forecasts:
        model = ModelLoader.load_model()
        if model is not None:
            try:
                future = PredictionManager.create_future_df(weather_forecasts)
                forecast = model.predict(future)
                metrics = PredictionManager.get_prediction_metrics(forecast)
                
                # Display results in a grid
                st.markdown("## 📊 Weekly Forecast")
                
                # Create two rows of columns
                row1_cols = st.columns(5)
                row2_cols = st.columns(3)
                all_cols = row1_cols + row2_cols
                
                for i, (metric, weather, col) in enumerate(zip(metrics, weather_forecasts, all_cols)):
                    with col:
                        st.markdown(f"### {metric['day']}")
                        st.markdown(f"**{metric['date'].strftime('%B %d, %Y')}**")
                        
                        # Weather info with icons
                        temp_icon = "🌡️" if weather['temperature'] > 20 else "❄️"
                        rain_icon = "🌧️" if weather['precipitation'] > 0 else "☀️"
                        
                        st.markdown(f"""
                            {temp_icon} **Temperature:** {weather['temperature']:.1f}°C  
                            {rain_icon} **Precipitation:** {weather['precipitation']:.1f}mm
                        """)
                        
                        # Holiday information with styled badge
                        if weather['holiday_label'] != 'NA':
                            st.markdown(f"""
                                <div class='holiday-badge'>
                                    🎉 {weather['holiday_label']}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.metric(
                            "Predicted Sales",
                            f"{metric['prediction']} loaves",
                            delta=f"±{metric['confidence_range']}"
                        )
                
                # Create visualization
                st.markdown("## 📈 Sales Trend")
                fig = go.Figure()
                
                # Add prediction points
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='markers+lines',
                    name='Prediction',
                    marker=dict(size=12, color='#1f77b4')
                ))
                
                # Add confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(31,119,180,0.2)'),
                    name='Confidence Interval'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='rgba(31,119,180,0.2)'),
                    showlegend=False
                ))
                
                fig.update_layout(
                    title='Weekly Sales Forecast Trend',
                    xaxis_title='Date',
                    yaxis_title='Number of Loaves',
                    showlegend=True,
                    height=500,
                    template='plotly_white',
                    margin=dict(t=50, b=50, l=50, r=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")

if __name__ == "__main__":
    main()