import pandas as pd#
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from preprocessing import create_preprocessor

#Load dataset
df = pd.read_csv("backend/data/StudentPerformanceFactors.csv")

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
    ("regressor", GradientBoostingRegressor(
        random_state=42
    ))
])

#params to test
param_grid = {
    "regressor__n_estimators": [100,200,300],
    "regressor__learning_rate": [0.03, 0.05, 0.1],
    "regressor__max_depth": [2, 3, 4]
}

#create gridsearch cv
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

#Train model
grid_search.fit(X_train, y_train)

#get best model
best_model = grid_search.best_estimator_

#get Feature Importance
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
importances = best_model.named_steps["regressor"].feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

#Predict
predictions = best_model.predict(X_test)

#model eval
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

#Output
print("Best Parameters:", grid_search.best_params_)
print("Best CV MAE:", -grid_search.best_score_)
print("Test MAE:", mae)
print("Test R² Score:", r2)

#saving trained model
joblib.dump(best_model, "backend/models/gradient_boosting_model.pkl")
print("Model Saved Successfully.")


