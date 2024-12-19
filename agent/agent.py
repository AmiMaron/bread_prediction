import openai
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from prophet import Prophet
from data.date_to_score import get_score_for_date

class ForecastAgent:
    def __init__(self, api_key: str, model: Prophet, weather_service: Any):
        """Initialize the forecast agent with necessary components"""
        self.api_key = api_key
        openai.api_key = api_key
        self.model = model
        self.weather_service = weather_service
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create the system prompt that defines the agent's behavior"""
        return """You are an expert AI assistant for a bakery in Paris, specializing in bread sales forecasting. 
        You have deep knowledge of the Prophet forecasting model and understand how various factors affect bread sales:

        Key factors in the forecasting model:
        1. Weather (temperature and precipitation)
        2. Day of the week
        3. Special events and holidays (scored 0-9, with 4 being a normal day)

        Your capabilities:
        1. Explain forecast results and their reasoning
        2. Handle custom forecast requests
        3. Provide insights about factors affecting bread sales

        Always be concise, practical, and focus on actionable insights for the bakery owner."""

    def explain_forecast(self, metrics: List[Dict], weather_forecasts: List[Dict]) -> str:
        """Generate an explanation for the current forecast"""
        # Prepare the forecast data for the prompt
        forecast_summary = self._prepare_forecast_summary(metrics, weather_forecasts)
        
        prompt = f"""Based on the following forecast data:
        {forecast_summary}
        
        Provide a brief, insightful analysis of:
        1. The overall trend for the week
        2. Any notable peaks or dips in demand
        3. Key factors influencing the predictions
        
        Keep the explanation concise and practical for the bakery owner."""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content

    def handle_custom_forecast(self, query: str) -> str:
        """Handle custom forecast requests from the user"""
        try:
            # Extract date from query using GPT-4
            date_extraction_prompt = f"""Extract the specific date or date range from the following query: '{query}'
            Return the response in the format: YYYY-MM-DD
            If a range is mentioned, return the start date only.
            If no specific date is mentioned, return 'NO_DATE'"""

            date_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a date extraction specialist."},
                    {"role": "user", "content": date_extraction_prompt}
                ]
            )

            extracted_date = date_response.choices[0].message.content.strip()
            
            if extracted_date == 'NO_DATE':
                return "I couldn't identify a specific date in your query. Please specify a date for the forecast."

            # Generate forecast for the extracted date
            target_date = datetime.strptime(extracted_date, '%Y-%m-%d').date()
            forecast_data = self._generate_custom_forecast(target_date)
            
            # Generate response
            response_prompt = f"""Based on the forecast data:
            {forecast_data}
            
            Provide a helpful response to the user's query: '{query}'
            Include specific numbers and relevant factors affecting the forecast."""

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": response_prompt}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"I encountered an error processing your request: {str(e)}"

    def _prepare_forecast_summary(self, metrics: List[Dict], weather_forecasts: List[Dict]) -> str:
        """Prepare a summary of the forecast data for the prompt"""
        summary = []
        for metric, weather in zip(metrics, weather_forecasts):
            summary.append(f"""
            Date: {metric['date']}
            Day: {metric['day']}
            Predicted Sales: {metric['prediction']} loaves (±{metric['confidence_range']})
            Temperature: {weather['temperature']:.1f}°C
            Precipitation: {weather['precipitation']:.1f}mm
            Event: {weather['holiday_label']}
            """)
        return "\n".join(summary)

    def _generate_custom_forecast(self, target_date: datetime.date) -> Dict:
        """Generate a forecast for a specific date"""
        # Get weather forecast
        weather_forecast = self.weather_service.get_weather_forecasts(48.8566, 2.3522, [target_date])
        
        if not weather_forecast:
            raise Exception("Unable to fetch weather forecast")

        # Create future dataframe
        future = pd.DataFrame([{
            'ds': pd.to_datetime(target_date),
            'temperature_2m_mean': weather_forecast[0]['temperature'],
            'precipitation_sum_mm': weather_forecast[0]['precipitation'],
            'holiday_score': weather_forecast[0]['holiday_score']
        }])

        # Add day dummies
        day_name = target_date.strftime('%A').lower()
        for day in ['friday', 'monday', 'saturday', 'sunday', 'thursday', 'tuesday', 'wednesday']:
            future[f'day_{day}'] = day == day_name

        # Generate forecast
        forecast = self.model.predict(future)
        
        return {
            'date': target_date,
            'prediction': int(forecast['yhat'].iloc[0]),
            'lower_bound': int(forecast['yhat_lower'].iloc[0]),
            'upper_bound': int(forecast['yhat_upper'].iloc[0]),
            'weather': weather_forecast[0]
        }

# Helper function to initialize the agent in the Streamlit app
def initialize_agent(model: Prophet, weather_service: Any) -> ForecastAgent:
    """Initialize the forecast agent with the necessary components"""
    return ForecastAgent(
        api_key=st.secrets["OPENAI_API_KEY"],
        model=model,
        weather_service=weather_service
    )