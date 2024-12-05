import streamlit as st
import pandas as pd
from prophet import Prophet
import datetime
import plotly.graph_objects as go
import pickle
import os

# Configure the Streamlit page
st.set_page_config(
    page_title="Bread Sales Predictor",
    page_icon="🥖",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained Prophet model"""
    try:
        model_path = os.path.join('model', 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_future_df(date, temperature, precipitation):
    """Create a DataFrame for prediction"""
    future = pd.DataFrame({
        'ds': [date],
        'temperature_2m_mean': [temperature],
        'precipitation_sum_mm': [precipitation]
    })
    
    # Add day dummies
    day_name = date.strftime('%A').lower()
    for day in ['friday', 'monday', 'saturday', 'sunday', 'thursday', 'tuesday', 'wednesday']:
        future[f'day_{day}'] = 1 if day == day_name else 0
    
    return future

def display_prediction_metrics(forecast):
    """Display the prediction metrics in columns"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Bread Loaves",
            value=f"{int(forecast['yhat'].iloc[0])}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Lower Bound",
            value=f"{int(forecast['yhat_lower'].iloc[0])}",
            delta=None
        )
        
    with col3:
        st.metric(
            label="Upper Bound",
            value=f"{int(forecast['yhat_upper'].iloc[0])}",
            delta=None
        )

def display_weather_info(date, temperature, precipitation, forecast):
    """Display weather and prediction information"""
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.info(f"Date: {date.strftime('%A, %B %d, %Y')}")
        st.info(f"Temperature: {temperature}°C")
        
    with info_col2:
        st.info(f"Precipitation: {precipitation}mm")
        confidence_range = f"±{int((forecast['yhat_upper'].iloc[0] - forecast['yhat_lower'].iloc[0])/2)} loaves"
        st.info(f"Confidence Range: {confidence_range}")

def main():
    st.title("🥖 Bread Sales Predictor")
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    
    # Date selector (defaulting to tomorrow)
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    date = st.sidebar.date_input("Select Date", value=tomorrow)
    
    # Temperature input
    temperature = st.sidebar.slider(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=40.0,
        value=20.0,
        step=0.5
    )
    
    # Precipitation input
    precipitation = st.sidebar.slider(
        "Precipitation (mm)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.5
    )
    
    # Load model
    model = load_model()
    
    if model is not None:
        try:
            # Create future DataFrame
            future = create_future_df(pd.to_datetime(date), temperature, precipitation)
            
            # Make prediction
            forecast = model.predict(future)
            
            # Display results
            st.header("Prediction Results")
            display_prediction_metrics(forecast)
            
            # Display weather info
            st.subheader("Prediction Details")
            display_weather_info(date, temperature, precipitation, forecast)
            
            # Historical context
            with st.expander("See Historical Context"):
                st.write("This prediction is based on historical sales data and takes into account:")
                st.write("- Day of the week patterns")
                st.write("- Temperature effects")
                st.write("- Precipitation impacts")
                st.write("- Seasonal trends")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    else:
        st.warning("Please ensure the model file is present in the 'model' directory.")

if __name__ == "__main__":
    main()