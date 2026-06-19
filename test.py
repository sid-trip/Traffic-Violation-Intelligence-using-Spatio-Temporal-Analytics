import pandas as pd

df = pd.read_csv("/home/siddhant/Traffic-Violation-Intelligence-using-Spatio-Temporal-Analytics/jan to may police violation_anonymized791b166.csv")

print(df["violation_type"].value_counts().head(20))
print(df["junction_name"].nunique())
print(df["police_station"].nunique())
print(df["created_datetime"].min())
print(df["created_datetime"].max())