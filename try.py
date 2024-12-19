from data.date_to_score import scorer

try_few_dates = ["2021/12/12", "24-12-2022", "25/12/2022", "01/08/2023"
]
result = [scorer.get_score(i) for i in try_few_dates]
print(result) 

