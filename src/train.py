import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from preprocessing import create_preprocessor

#Load dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

x = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

#splitting data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
#creating preprocessing system
preprocessor = create_preprocessor()

#process traaaining data
x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

#Create model
model = LinearRegression()
#Train Model
model.fit(x_train_processed, y_train)

#Predict
predictions=model.predict(x_test_processed)
mae = mean_absolute_error(y_test, predictions)

#Output
print("Mean Absolute Error", mae)


