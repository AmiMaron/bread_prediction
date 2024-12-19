from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import holidays
import openai
import pandas as pd
from pydantic import BaseModel, Field
import toml
from loguru import logger
import sys
import os

class Config:
    """Configuration management class."""
    def __init__(self, secrets_path: Path):
        self.secrets_path = secrets_path
        self.api_key: Optional[str] = None
        self.model_name: str = "gpt-4o-mini"
        self.temperature: float = 0.0

    def load_secrets(self) -> None:
        """Load secrets from TOML file."""
        try:
            secrets = toml.load(".streamlit/secrets.toml")
            self.api_key = secrets.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("API key not found in secrets file")
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            raise

class EventScoring(BaseModel):
    """Pydantic model for event scoring response."""
    score: int
    
class EventManager:
    """Manages event-related operations."""
    def __init__(self):
        self.events: Dict[date, str] = self._initialize_events()

    def _initialize_events(self) -> Dict[date, str]:
        """Initialize event dictionary with holidays and custom events."""
        current_year = date.today().year
        french_holidays = holidays.FR(years=range(2021, current_year + 2))
        
        custom_events = {
        # --- Chandeleur (Candlemas/Crêpe Day) - February 2 each year ---
        date(2021, 2, 2): "Chandeleur (Candlemas)",
        date(2022, 2, 2): "Chandeleur (Candlemas)",
        date(2023, 2, 2): "Chandeleur (Candlemas)",
        date(2024, 2, 2): "Chandeleur (Candlemas)",
        date(2025, 2, 2): "Chandeleur (Candlemas)",

        # --- Saint-Valentin (Valentine’s Day) - February 14 each year ---
        date(2021, 2, 14): "Saint-Valentin (Valentine’s Day)",
        date(2022, 2, 14): "Saint-Valentin (Valentine’s Day)",
        date(2023, 2, 14): "Saint-Valentin (Valentine’s Day)",
        date(2024, 2, 14): "Saint-Valentin (Valentine’s Day)",
        date(2025, 2, 14): "Saint-Valentin (Valentine’s Day)",

        # --- Mardi Gras (Carnaval) - 47 days before Easter Sunday ---
        # Example dates provided: Feb 16, 2021 and Feb 21, 2023
        date(2021, 2, 16): "Mardi Gras (Carnaval)",
        date(2023, 2, 21): "Mardi Gras (Carnaval)",
        date(2024, 2, 13): "Mardi Gras (Carnaval)",
        date(2025, 3, 4): "Mardi Gras (Carnaval)",

        # --- Beaujolais Nouveau Day - The third Thursday of November ---
        # Example dates provided: Nov 18, 2021 and Nov 16, 2023
        date(2021, 11, 18): "Beaujolais Nouveau Day",
        date(2023, 11, 16): "Beaujolais Nouveau Day",
        date(2024, 11, 21): "Beaujolais Nouveau Day",
        date(2025, 11, 20): "Beaujolais Nouveau Day",
        }
        
        return {**french_holidays, **custom_events}

    def get_event_description(self, target_date: date) -> str:
        """Get event description for a given date."""
        if target_date in self.events:
            return self.events[target_date]
        
        next_day = target_date + timedelta(days=1)
        if next_day in self.events:
            return f"Eve of {self.events[next_day]}"
        
        return "No significant events"

class OpenAIClient:
    """Handles OpenAI API interactions."""
    def __init__(self, config: Config):
        self.config = config
        openai.api_key = config.api_key

    def get_event_score(self, date_str: str, event_description: str) -> Optional[EventScoring]:
        """Get event impact score from OpenAI."""
        system_prompt = {
            "role": "system",
            "content": (
                "You are a helpful AI that provides impact scores for special events. "
                "Return valid JSON that fits the Pydantic schema: {score: int}."
            )
        }

        user_prompt = {
            "role": "user",
            "content": self._create_scoring_prompt(date_str, event_description)
        }

        try:
            completion = openai.beta.chat.completions.parse(
                model=self.config.model_name,
                messages=[system_prompt, user_prompt],
                response_format=EventScoring,
                temperature=self.config.temperature
            )
            return EventScoring.model_validate_json(completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI API error for date {date_str}: {e}")
            return None

    @staticmethod
    def _create_scoring_prompt(date_str: str, event_description: str) -> str:
        """Create the scoring prompt for OpenAI."""
        return f"""I am working on a machine learning project to predict demand for a bakery in France. 
        I want to assign an event impact score (0–9) for specific dates based on how much each event (or event eve) 
        is likely to influence demand.

        Scoring Guidelines:
        0-3: Events with negative impact on demand (such as days with public transport strikes)
        4: Default score for days with no impactful events (regular days)
        5-7: Events with moderate to significant impact (such as Valentine's Day)
        8-9: Major events with very high impact on demand (such as Christmas Eve & Christmas Day)

        Date: {date_str}
        Event Description: {event_description}

        Provide your response as JSON that satisfies this structure:
        {{
          "score": <integer from 0 to 9>
        }}"""

class EventProcessor:
    """Main processor for event scoring."""
    def __init__(self, config: Config):
        self.event_manager = EventManager()
        self.openai_client = OpenAIClient(config)

    def _parse_date(self, date_str: str) -> date:
        """Parse date string in multiple formats."""
        formats = [
            "%d/%m/%Y",  # DD/MM/YYYY
            "%d-%m-%Y",  # DD-MM-YYYY
            "%Y-%m-%d",  # YYYY-MM-DD
            "%Y/%m/%d"   # YYYY/MM/DD
        ]
        
        for date_format in formats:
            try:
                return datetime.strptime(date_str, date_format).date()
            except ValueError:
                continue
        
        raise ValueError(f"Date string '{date_str}' does not match any supported format")

    def process_single_date(self, date_str: str) -> Dict[str, Any]:
        """Process a single date and return event information."""
        try:
            # Parse the date string in DD/MM/YYYY format
            date_obj = self._parse_date(date_str)
            event_description = self.event_manager.get_event_description(date_obj)
            scoring_result = self.openai_client.get_event_score(date_str, event_description)
            score = scoring_result.score if scoring_result else 4

            return {
                "date": date_str,
                "event_description": event_description,
                "score": score
            }
        except ValueError as e:
            logger.error(f"Error processing date {date_str}: {e}")
            return {"error": str(e)}

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a DataFrame of dates and add event information."""
        if "date" not in df.columns:
            raise ValueError("Input DataFrame must have a 'date' column")

        df = df.copy()
        df["event_description"] = "No significant events"
        df["event_score"] = 4

        logger.info(f"Processing {len(df)} rows...")
        for idx, row in df.iterrows():
            result = self.process_single_date(row["date"])
            if "error" not in result:
                df.at[idx, "event_description"] = result["event_description"]
                df.at[idx, "event_score"] = result["score"]

        return df

def main():
    """Main entry point for the script."""
    # Setup logging
    logger.add("event_scoring.log", rotation="500 MB")

    try:
        # Initialize configuration
        config = Config(Path("secrets.toml"))
        config.load_secrets()

        # Initialize processor
        processor = EventProcessor(config)

        PROCESS_CSV = True
        # Process CSV if requested
        if PROCESS_CSV:
            print("Processing CSV file...")
            input_path = r"c:/Users/avrahamma/Documents/School/AI_for_social_good/data/sales_and_weather_data.csv"
            output_path = r"c:/Users/avrahamma/Documents/School/AI_for_social_good/data/sales_weather_and_score_data.csv"

            df = pd.read_csv(input_path)
            processed_df = processor.process_dataframe(df)
            processed_df.to_csv(output_path, index=False)
            logger.success(f"Output saved to {output_path}")

    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


class EventScoringAPI:
    """Simple interface for event scoring with automatic configuration."""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.processor = None
            self._initialize()
            EventScoringAPI._initialized = True
    
    def _get_api_key(self) -> str:
        """Get API key from various possible sources."""
        # Try environment variable first
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
            
        # Try common config file locations
        config_locations = [
            Path.home() / ".streamlit" / "secrets.toml",
            Path("secrets.toml"),
            Path(".streamlit/secrets.toml")
        ]
        
        for config_path in config_locations:
            if config_path.exists():
                try:
                    secrets = toml.load(config_path)
                    if "OPENAI_API_KEY" in secrets:
                        return secrets["OPENAI_API_KEY"]
                except:
                    continue
                    
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
            "or include it in a secrets.toml file."
        )
    
    def _initialize(self):
        """Initialize the scoring system automatically."""
        class AutoConfig(Config):
            def __init__(self):
                self.model_name = "gpt-4o-mini"
                self.temperature = 0.0
                self.api_key = None
            
            def load_secrets(self):
                pass  # Not needed as we handle it in EventScoringAPI
        
        config = AutoConfig()
        config.api_key = self._get_api_key()
        self.processor = EventProcessor(config)
    
    def get_score(self, date_str: str) -> dict:
        """
        Get the event score for a specific date.
        
        Args:
            date_str: Date string in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary containing date info, event description, and score
        """
        return self.processor.process_single_date(date_str)

# Create a global instance
scorer = EventScoringAPI()

