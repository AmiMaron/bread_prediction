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
    
    def get_weather_forecast(self, lat: float, lon: float, target_date: datetime.date) -> Optional[Dict]:
        """
        Get weather forecast for specific date and location
        Returns temperature (°C) and precipitation (mm)
        """
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'  # Get temperature in Celsius
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            forecast_data = response.json()
            
            # Find forecast closest to target date
            target_datetime = datetime.datetime.combine(target_date, datetime.time(12, 0))  # Noon on target date
            
            closest_forecast = min(
                forecast_data['list'],
                key=lambda x: abs(datetime.datetime.fromtimestamp(x['dt']) - target_datetime)
            )
            
            return {
                'temperature': closest_forecast['main']['temp'],
                'precipitation': closest_forecast.get('rain', {}).get('3h', 0)  # mm per 3h
            }
            
        except Exception as e:
            st.error(f"Error fetching weather data: {str(e)}")
            return None

class PredictionManager:
    @staticmethod
    def create_future_df(date: datetime.date, weather_data: Dict) -> pd.DataFrame:
        """Create a DataFrame for prediction using weather forecast data"""
        future = pd.DataFrame({
            'ds': [pd.to_datetime(date)],
            'temperature_2m_mean': [weather_data['temperature']],
            'precipitation_sum_mm': [weather_data['precipitation'] * 8]  # Convert 3h to daily
        })
        
        # Add day dummies
        day_name = date.strftime('%A').lower()
        day_columns = ['friday', 'monday', 'saturday', 'sunday', 'thursday', 'tuesday', 'wednesday']
        future.loc[:, [f'day_{day}' for day in day_columns]] = [1 if day == day_name else 0 for day in day_columns]
        
        return future

    @staticmethod
    def get_prediction_metrics(forecast: pd.DataFrame) -> dict:
        """Extract and format prediction metrics from the forecast"""
        return {
            'prediction': int(forecast['yhat'].iloc[0]),
            'lower_bound': int(forecast['yhat_lower'].iloc[0]),
            'upper_bound': int(forecast['yhat_upper'].iloc[0]),
            'confidence_range': int((forecast['yhat_upper'].iloc[0] - forecast['yhat_lower'].iloc[0])/2)
        }
    
def main():
    st.set_page_config(
        page_title="Bread Sales Predictor",
        page_icon="🥖",
        layout="wide"
    )
    
    # Initialize services
    weather_service = WeatherService(st.secrets["OPENWEATHER_API_KEY"])
    
    st.title("🥖 Bread Sales Predictor")
    
    # Bakery location (could be made configurable)
    PARIS_LAT = 48.8566
    PARIS_LON = 2.3522
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    
    # Date selector (limited to 5 days ahead due to free API limitations)
    max_date = datetime.date.today() + datetime.timedelta(days=5)
    date = st.sidebar.date_input(
        "Select Date",
        value=datetime.date.today() + datetime.timedelta(days=1),
        min_value=datetime.date.today() + datetime.timedelta(days=1),
        max_value=max_date
    )
    
    # Fetch weather forecast
    weather_data = weather_service.get_weather_forecast(PARIS_LAT, PARIS_LON, date)
    
    if weather_data:
        # Display weather forecast
        st.sidebar.subheader("Weather Forecast")
        st.sidebar.info(f"Temperature: {weather_data['temperature']:.1f}°C")
        st.sidebar.info(f"Precipitation: {weather_data['precipitation']:.1f}mm/3h")
        
        # Allow manual override
        use_forecast = st.sidebar.checkbox("Use weather forecast", value=True)
        if not use_forecast:
            weather_data['temperature'] = st.sidebar.slider(
                "Temperature (°C)",
                min_value=-10.0,
                max_value=40.0,
                value=float(weather_data['temperature']),
                step=0.5
            )
            weather_data['precipitation'] = st.sidebar.slider(
                "Precipitation (mm/3h)",
                min_value=0.0,
                max_value=50.0,
                value=float(weather_data['precipitation']),
                step=0.5
            )
        
        # Load model and make prediction
        model = ModelLoader.load_model()
        if model is not None:
            try:
                future = PredictionManager.create_future_df(date, weather_data)
                forecast = model.predict(future)
                metrics = PredictionManager.get_prediction_metrics(forecast)
                
                # Display results in main area
                st.header("Prediction Results")
                
                # Key metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Predicted Bread Loaves",
                        value=str(metrics['prediction']),
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Lower Bound",
                        value=str(metrics['lower_bound']),
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Upper Bound",
                        value=str(metrics['upper_bound']),
                        delta=None
                    )
                
                # Create visualization
                fig = go.Figure()
                
                # Add prediction point
                fig.add_trace(go.Scatter(
                    x=[forecast['ds'].iloc[0]],
                    y=[forecast['yhat'].iloc[0]],
                    mode='markers',
                    name='Prediction',
                    marker=dict(size=12, color='blue')
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=[forecast['ds'].iloc[0], forecast['ds'].iloc[0]],
                    y=[forecast['yhat_lower'].iloc[0], forecast['yhat_upper'].iloc[0]],
                    mode='lines',
                    name='Confidence Interval',
                    line=dict(color='rgba(0,0,255,0.2)', width=2)
                ))
                
                fig.update_layout(
                    title='Bread Sales Prediction',
                    xaxis_title='Date',
                    yaxis_title='Number of Loaves',
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed information
                st.subheader("Prediction Details")
                
                # Create two columns for details
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.info(f"Date: {date.strftime('%A, %B %d, %Y')}")
                    st.info(f"Temperature: {weather_data['temperature']:.1f}°C")
                
                with detail_col2:
                    st.info(f"Precipitation: {weather_data['precipitation']:.1f}mm/3h")
                    st.info(f"Confidence Range: ±{metrics['confidence_range']} loaves")
                
                # Additional context
                with st.expander("See Prediction Context"):
                    st.write("""
                    This prediction takes into account:
                    - Historical sales patterns
                    - Day of the week effects
                    - Weather forecast data
                    - Seasonal trends
                    
                    The confidence range indicates the uncertainty in the prediction.
                    Consider preparing extra inventory if you have upcoming events or promotions.
                    """)
                
                # Recommendations
                st.subheader("Recommendations")
                
                # Calculate buffer based on confidence range
                buffer = round(metrics['confidence_range'] * 0.2)  # 20% of confidence range
                
                st.success(f"""
                📊 **Suggested Production Plan:**
                - Base prediction: {metrics['prediction']} loaves
                - Recommended buffer: +{buffer} loaves
                - Total recommended production: {metrics['prediction'] + buffer} loaves
                
                This includes a small buffer to account for prediction uncertainty and potential unexpected demand.
                """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please try different input parameters")

if __name__ == "__main__":
    main()