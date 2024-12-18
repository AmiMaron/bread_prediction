# import functions from holiday_impact_generation.py
from data.holiday_impact_generation import get_score_for_date

try_few_dates = ["2024-11-21", "2025-11-20", "2026-11-19", "2027-11-18", "2028-11-16", "2021-01-01"]
# a = [get_score_for_date(i) for i in try_few_dates]
# print(a)
score = get_score_for_date("2024-11-21")
print(list(score.values())[2])