import pandas as pd
# from preprocessing import preprocess_data

#Load dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

#Load data types of the dataset
for column in df.select_dtypes(include="object").columns:
    print(f"\n{column}:")
    print(df[column].unique())

