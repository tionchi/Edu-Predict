import pandas as pd
from preprocessing import preprocess_data

#Load dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

processed_df = preprocess_data(df)

print(processed_df.select_dtypes(include="object").columns.tolist())