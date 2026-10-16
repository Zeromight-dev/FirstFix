import pandas as pd
loc = pd.read_csv("data/locations.csv")
inc = pd.read_csv("data/incidents.csv")
com = pd.read_csv("data/public_complaints.csv")

for name, df in [("locations", loc), ("incidents", inc), ("complaints", com)]:
    print("=====", name, df.shape)
    print(df.dtypes)
    print(df.head(3).to_string())
    print("missing:\n", df.isna().sum()[df.isna().sum() > 0])
    print("duplicates:", df.duplicated().sum())

print("location_ids not in locations:",
      (~inc.location_id.isin(loc.location_id)).sum(),
      (~com.location_id.isin(loc.location_id)).sum())