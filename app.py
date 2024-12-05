import streamlit as st
import pandas as pd
from prophet import Prophet
import datetime
import plotly.graph_objects as go
import pickle
import os
from typing import Optional, Tuple

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

class DataValidator:
    @staticmethod
    def validate_inputs(date: datetime.date, temperature: float, precipitation: float) -> Tuple[bool, str]:
        """Validate user inputs"""
        if date < datetime.date.today():
            return False, "Please select a future date"
        
        if not -10 <= temperature <= 40:
            return False, "Temperature must be between -10°C and 40°C"
            
        if not 0 <= precipitation <= 50:
            return False, "Precipitation must be between 0mm and 50mm"
            
        return True, ""

class PredictionManager:
    @staticmethod
    def create_future_df(date: datetime.date, temperature: float, precipitation: float) -> pd.DataFrame:
        """Create a DataFrame for prediction with input validation"""
        future = pd.DataFrame({
            'ds': [pd.to_datetime(date)],
            'temperature_2m_mean': [temperature],
            'precipitation_sum_mm': [precipitation]
        })
        
        # Add day dummies with vectorized operations
        day_name = date.strftime('%A').lower()
        day_columns = ['friday', 'monday', 'saturday', 'sunday', 'thursday', 'tuesday', 'wednesday']
        future.loc[:, [f'day_{day}' for day in day_columns]] = [1 if day == day_name else 0 for day in day_columns]
        
        return future

    @staticmethod
    def get_prediction_metrics(forecast: pd.DataFrame) -> dict:
        """Extract and format prediction metrics"""
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
    
    st.title("🥖 Bread Sales Predictor")
    
    # Initialize components
    model_loader = ModelLoader()
    data_validator = DataValidator()
    prediction_manager = PredictionManager()
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    date = st.sidebar.date_input("Select Date", value=tomorrow)
    
    temperature = st.sidebar.slider(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=40.0,
        value=20.0,
        step=0.5
    )
    
    precipitation = st.sidebar.slider(
        "Precipitation (mm)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.5
    )
    
    # Validate inputs
    is_valid, error_message = data_validator.validate_inputs(date, temperature, precipitation)
    if not is_valid:
        st.error(error_message)
        return
    
    # Load model
    model = model_loader.load_model()
    if model is None:
        return
    
    try:
        # Create future DataFrame and make prediction
        future = prediction_manager.create_future_df(date, temperature, precipitation)
        forecast = model.predict(future)
        metrics = prediction_manager.get_prediction_metrics(forecast)
        
        # Display results
        st.header("Prediction Results")
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Bread Loaves", str(metrics['prediction']))
        col2.metric("Lower Bound", str(metrics['lower_bound']))
        col3.metric("Upper Bound", str(metrics['upper_bound']))
        
        # Create and display plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[forecast['ds'].iloc[0]],
            y=[forecast['yhat'].iloc[0]],
            mode='markers',
            name='Prediction',
            marker=dict(size=12, color='blue')
        ))
        
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
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display prediction details
        st.subheader("Prediction Details")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.info(f"Date: {date.strftime('%A, %B %d, %Y')}")
            st.info(f"Temperature: {temperature}°C")
        with info_col2:
            st.info(f"Precipitation: {precipitation}mm")
            st.info(f"Confidence Range: ±{metrics['confidence_range']} loaves")
        
        # Historical context
        with st.expander("See Historical Context"):
            st.write("This prediction is based on historical sales data and takes into account:")
            st.write("- Day of the week patterns")
            st.write("- Temperature effects")
            st.write("- Precipitation impacts")
            st.write("- Seasonal trends")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.error("Please try again with different input parameters")

if __name__ == "__main__":
    main()