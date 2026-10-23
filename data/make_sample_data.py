"""Generate a synthetic 'Superstore-style' dataset + a benchmark with correct answers.

    python data/make_sample_data.py

Creates data/sample_sales.csv and eval/questions.json. Answers are computed with pandas, so they are
guaranteed correct. To use a real Kaggle dataset instead, skip this script and write your own questions.
"""
import json
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000
regions = ["West", "East", "Central", "South"]
cats = {"Technology": ["Phones", "Laptops", "Accessories"],
        "Furniture": ["Chairs", "Tables", "Bookcases"],
        "Office Supplies": ["Paper", "Binders", "Storage"]}
margin = {"Technology": 0.18, "Furniture": 0.05, "Office Supplies": 0.12}
price = {"Technology": 450, "Furniture": 300, "Office Supplies": 60}

rows = []
for i in range(N):
    cat = rng.choice(list(cats), p=[0.25, 0.30, 0.45])
    sub = rng.choice(cats[cat])
    date = pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 4 * 365)))
    # sales grow a little every year so year-over-year questions are interesting
    growth = 1 + 0.12 * (date.year - 2015)
    sales = round(float(rng.gamma(2.0, price[cat] / 2) * growth), 2)
    discount = float(rng.choice([0, 0.1, 0.2, 0.3, 0.4], p=[0.4, 0.25, 0.2, 0.1, 0.05]))
    profit = round(sales * (margin[cat] - discount * 0.6 + rng.normal(0, 0.05)), 2)
    rows.append({"order_id": f"ORD-{i + 1:05d}", "order_date": date.strftime("%Y-%m-%d"),
                 "region": rng.choice(regions, p=[0.3, 0.28, 0.24, 0.18]),
                 "segment": rng.choice(["Consumer", "Corporate", "Home Office"], p=[0.5, 0.3, 0.2]),
                 "category": cat, "sub_category": sub, "sales": sales,
                 "quantity": int(rng.integers(1, 9)), "discount": discount, "profit": profit})

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/sample_sales.csv", index=False)

d = df.assign(year=df.order_date.str[:4].astype(int), month=df.order_date.str[5:7].astype(int))
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
fmt0 = lambda x: str(int(round(x)))
fmt2 = lambda x: f"{x:.2f}"
cat_by_year = d.groupby(["category", "year"]).sales.sum().unstack()

Q = [
    ("Which region has the highest total profit?", d.groupby("region").profit.sum().idxmax()),
    ("Which product category has the lowest total sales?", d.groupby("category").sales.sum().idxmin()),
    ("How many orders were placed in 2017?", fmt0((d.year == 2017).sum())),
    ("What is the total sales amount in 2018?", fmt0(d[d.year == 2018].sales.sum())),
    ("Which customer segment has the highest average profit per order?", d.groupby("segment").profit.mean().idxmax()),
    ("Which sub-category has the highest total quantity sold?", d.groupby("sub_category").quantity.sum().idxmax()),
    ("What is the average discount across all orders? Give it as a decimal.", fmt2(d.discount.mean())),
    ("Which year had the highest total profit?", str(d.groupby("year").profit.sum().idxmax())),
    ("Which region has the lowest average discount?", d.groupby("region").discount.mean().idxmin()),
    ("How many orders had a negative profit?", fmt0((d.profit < 0).sum())),
    ("What is the total profit of the Technology category?", fmt0(d[d.category == "Technology"].profit.sum())),
    ("Which month of 2018 had the highest total sales? Answer with the month name.",
     months[int(d[d.year == 2018].groupby("month").sales.sum().idxmax()) - 1]),
    ("What is the average sales value of an order in the West region?", fmt2(d[d.region == "West"].sales.mean())),
    ("Which category has the highest profit margin (total profit divided by total sales)?",
     (d.groupby("category").profit.sum() / d.groupby("category").sales.sum()).idxmax()),
    ("How many orders were placed in the Consumer segment?", fmt0((d.segment == "Consumer").sum())),
    ("Which region had the most orders in 2016?", d[d.year == 2016].region.value_counts().idxmax()),
    ("What is the highest single-order sales value?", fmt2(d.sales.max())),
    ("Which category had the largest increase in total sales from 2017 to 2018?",
     (cat_by_year[2018] - cat_by_year[2017]).idxmax()),
    ("What percentage of total sales comes from the East region? Give one decimal place.",
     f"{100 * d[d.region == 'East'].sales.sum() / d.sales.sum():.1f}"),
    ("How many distinct sub-categories are there?", fmt0(d.sub_category.nunique())),
]
os.makedirs("eval", exist_ok=True)
with open("eval/questions.json", "w") as f:
    json.dump([{"id": i + 1, "question": q, "answer": str(a)} for i, (q, a) in enumerate(Q)], f, indent=2)
print(f"Wrote data/sample_sales.csv ({len(df)} rows) and eval/questions.json ({len(Q)} questions)")
