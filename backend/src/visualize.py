import joblib
import pandas as pd
import matplotlib.pyplot as plt

#load trained model
model = joblib.load("backend/models/student_performance_model.pkl")


#get feature names and importance
preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["regressor"]

feature_names = preprocessor.get_feature_names_out()
importances = regressor.feature_importances_

#create dataframe
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

#CLean feature names
feature_importance["Feature"] = (
    feature_importance["Feature"]
    .str.replace("ordinal__", "", regex=False)
    .str.replace("binary__", "", regex=False)
    .str.replace("nominal__", "", regex=False)
    .str.replace("numeric__", "", regex=False)
    .str.replace("_Yes", "", regex=False)
    .str.replace("_No", "", regex=False)
    .str.replace("_Male", "", regex=False)
    .str.replace("_Female", "", regex=False)
    .str.replace("_Public", "", regex=False)
    .str.replace("_Private", "", regex=False)
    .str.replace("_Negative", "", regex=False)
    .str.replace("_Neutral", "", regex=False)
    .str.replace("_Positive", "", regex=False)
)

#combine duplicate features
feature_importance = (
    feature_importance
    .groupby("Feature", as_index=False)["Importance"]
    .sum()
    .sort_values("Importance", ascending=True)
)

#results
print(feature_importance)

#create chart
plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("EduPredict Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300)

plt.show()