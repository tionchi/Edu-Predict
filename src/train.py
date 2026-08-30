import pandas as pd#
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from preprocessing import create_preprocessor

#Load dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

df = df[df["Exam_Score"].between(0, 100)]

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

#splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#creating preprocessing system
preprocessor = create_preprocessor()

#Create ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

#Train model
model.fit(X_train, y_train)

#Predict
predictions = model.predict(X_test)

#model eval
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

#Output
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
print(y.describe())

#saving trained model
joblib.dump(model, "models/student_performance_model.pkl")
print("Model Saved Successfully.")


