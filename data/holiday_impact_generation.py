import openai
from datetime import date
import holidays
import pandas as pd
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional
import toml

# ===== STEP 1: DEFINE OUR Pydantic MODEL FOR STRUCTURED OUTPUT =====
class HolidayScoring(BaseModel):
    score: int

# ===== STEP 2: GET HOLIDAY INFO =====
def get_holiday_info(date_obj, holiday_dict):
    """
    Checks if the given date is a holiday or the eve of a holiday in the provided holiday dictionary.
    Returns a tuple (holiday_name, is_eve).
    """
    # Check if the date itself is a holiday
    if date_obj in holiday_dict:
        return (holiday_dict[date_obj], False)

    # Check if it's the day before a holiday (holiday eve)
    next_day = date_obj + timedelta(days=1)
    if next_day in holiday_dict:
        return (f"Eve of {holiday_dict[next_day]}", True)

    return (None, False)

# ===== STEP 3: CALL OPENAI WITH STRUCTURED OUTPUTS =====
def holiday_impact_scoring(date_str, holiday_label=None):
    """
    Makes a call to OpenAI using structured outputs to get a holiday impact score (0–4).
    Now also passes holiday information to ChatGPT (holiday_label) for more context.
    Returns a HolidayScoring object or None if there's an error.
    """

    system_instructions = {
        "role": "system",
        "content": (
            "You are a helpful AI that provides holiday impact scores for bread demand. "
            "Return valid JSON that fits the Pydantic schema: {score: int}."
        )
    }

    # Build the prompt to include holiday info if provided
    holiday_context = (
        f"Holiday: {holiday_label}"
        if holiday_label and holiday_label != "NA"
        else "Holiday: None"
    )

    user_prompt = f"""I am working on a machine learning project to predict bread sales for a bakery in France. 
I want to assign a holiday impact score (0–4) for specific dates based on how much each holiday (or holiday eve) 
is likely to influence bread demand.

Scoring Guidelines:
0: Regular day with no holiday significance
1: Minor holidays or events with minimal impact on bread demand
2: Moderate-impact holidays that might increase sales slightly
3: High-impact holidays with noticeable bread sales increases
4: Very high-impact holidays or holiday eves with peak bread demand

Date: {date_str}
{holiday_context}

Provide your response as JSON that satisfies this structure:
{{
  "score": <integer from 0 to 4>
}}

Do NOT include any additional keys besides 'score'.
"""

    try:
        completion = openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[system_instructions, {"role": "user", "content": user_prompt}],
            response_format=HolidayScoring,
            temperature=0.0
        )
        structured_response: HolidayScoring = completion.choices[0].message.content
        return HolidayScoring.model_validate_json(structured_response)    
    except Exception as e:
        print(f"Error calling OpenAI for date {date_str}: {e}")
        return None

def main():
    # Load your API key
    secrets = toml.load(r"c:/Users/avrahamma/Documents/School/AI_for_social_good/.streamlit/secrets.toml")
    openai.api_key = secrets["OPENAI_API_KEY"]
    print("Loaded OpenAI API key successfully.")

    # Define the holiday dictionary
    current_year = date.today().year
    # Include relevant years for your data
    french_holidays = holidays.FR(years=range(2021, current_year + 2))

    # Manually added relevant or custom holiday dates
    custom_dates = {
    # --- Chandeleur (Candlemas/Crêpe Day) - February 2 each year ---
    date(2021, 2, 2): "Chandeleur (Candlemas)",
    date(2022, 2, 2): "Chandeleur (Candlemas)",
    date(2023, 2, 2): "Chandeleur (Candlemas)",

    # --- Saint-Valentin (Valentine’s Day) - February 14 each year ---
    date(2021, 2, 14): "Saint-Valentin (Valentine’s Day)",
    date(2022, 2, 14): "Saint-Valentin (Valentine’s Day)",
    date(2023, 2, 14): "Saint-Valentin (Valentine’s Day)",

    # --- Mardi Gras (Carnaval) - 47 days before Easter Sunday ---
    # Example dates provided: Feb 16, 2021 and Feb 21, 2023
    date(2021, 2, 16): "Mardi Gras (Carnaval)",
    date(2023, 2, 21): "Mardi Gras (Carnaval)",

    # --- Beaujolais Nouveau Day - The third Thursday of November ---
    # Example dates provided: Nov 18, 2021 and Nov 16, 2023
    date(2021, 11, 18): "Beaujolais Nouveau Day",
    date(2023, 11, 16): "Beaujolais Nouveau Day",
}
    
    for d, name in custom_dates.items():
        french_holidays[d] = name

    # Read the input CSV
    input_path = r"c:\Users\avrahamma\Documents\School\AI_for_social_good\data\paris_bread_sales.csv"
    print(f"Reading CSV from: {input_path}")
    df = pd.read_csv(input_path)

    if "date" not in df.columns:
        raise ValueError("Input CSV must have a 'date' column in YYYY-MM-DD or DD/MM/YYYY format.")

    # Prepare columns for holiday info and scoring
    df["holiday_name"] = "NA"
    df["holiday_score"] = 0

    print(f"Processing {len(df)} rows...")
    for i, row in df.iterrows():
        date_str = str(row["date"])
        try:
            # Adjust as needed for your CSV format (sample uses DD/MM/YYYY)
            date_obj = datetime.strptime(date_str, "%d/%m/%Y").date()
        except ValueError:
            print(f"Skipping invalid date format: {date_str}")
            continue

        # Identify holiday or holiday eve
        holiday_name, is_eve = get_holiday_info(date_obj, french_holidays)
        combined_holiday_label = "NA"
        if holiday_name:
            combined_holiday_label = holiday_name
        if is_eve:
            # If it's an eve, append "(Eve)"
            combined_holiday_label += " (Eve)" if holiday_name else "Eve"

        df.at[i, "holiday_name"] = combined_holiday_label

        print(f"Row {i}: Date {date_str}, Holiday: {combined_holiday_label}")

        # Pass both the date and holiday info to ChatGPT for scoring
        scoring_result = holiday_impact_scoring(date_str, combined_holiday_label)
        if scoring_result is not None:
            print(f" - Score received: {scoring_result.score}")
            df.at[i, "holiday_score"] = scoring_result.score
        else:
            print(" - No score returned due to error or parsing failure.")

    output_path = "holiday_scored.csv"
    df.to_csv(output_path, index=False)
    print(f"Output saved to {output_path}")

# if __name__ == "__main__":
    # main()

# add holiday_name and holiday_score to "paris_bread_sales.csv"
def merge_holiday_data(sales_file, holiday_file, output_file):
    # Read the sales and holiday data
    sales_df = pd.read_csv(sales_file)
    holiday_df = pd.read_csv(holiday_file, usecols=['date', 'holiday_name', 'holiday_score'])

    # Merge the data on the 'date' column
    merged_df = pd.merge(sales_df, holiday_df, on='date', how='left')

    # Reorder columns to ensure holiday_name and holiday_score appear at the end
    cols = [col for col in merged_df.columns if col not in ['holiday_name', 'holiday_score']]
    cols += ['holiday_name', 'holiday_score']
    merged_df = merged_df[cols]

    # Save the merged data to a new file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved as {output_file}")

# File paths
sales_file = r"c:\Users\avrahamma\Documents\School\AI_for_social_good\data\paris_bread_sales.csv"
holiday_file = r"c:\Users\avrahamma\Documents\School\AI_for_social_good\data\holiday_scored.csv"
output_file = r"c:\Users\avrahamma\Documents\School\AI_for_social_good\data\paris_bread_sales_with_holidays.csv"

# Perform the merge
merge_holiday_data(sales_file, holiday_file, output_file)



