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
        # Change the model path to match your GitHub structure
        model_path = os.path.join('model', 'prophet_model.pkl')
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
    day_columns = ['friday', 'monday', 'saturday', 'sunday', 'thursday', 'tuesday', 'wednesday']
    for day in day_columns:
        future[f'day_{day}'] = 1 if day == day_name else 0
    
    return future

def plot_prediction(forecast):
    """Create a plotly figure for the prediction"""
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
    
    return fig

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
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Bread Loaves", f"{int(forecast['yhat'].iloc[0])}")
            with col2:
                st.metric("Lower Bound", f"{int(forecast['yhat_lower'].iloc[0])}")
            with col3:
                st.metric("Upper Bound", f"{int(forecast['yhat_upper'].iloc[0])}")
            
            # Plot the prediction
            st.plotly_chart(plot_prediction(forecast), use_container_width=True)
            
            # Display weather info
            st.subheader("Prediction Details")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.info(f"Date: {date.strftime('%A, %B %d, %Y')}")
                st.info(f"Temperature: {temperature}°C")
            with info_col2:
                st.info(f"Precipitation: {precipitation}mm")
                confidence_range = f"±{int((forecast['yhat_upper'].iloc[0] - forecast['yhat_lower'].iloc[0])/2)} loaves"
                st.info(f"Confidence Range: {confidence_range}")
            
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