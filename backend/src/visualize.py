import joblib
import pandas as pd
import matplotlib.pyplot as plt


models = {
    "Linear Regression": (
        "backend/models/linear_regression_model.pkl",
        "linear_regression_feature_importance.png",
    ),
    "Random Forest": (
        "backend/models/random_forest_model.pkl",
        "random_forest_feature_importance.png",
    ),
    "Gradient Boosting": (
        "backend/models/gradient_boosting_model.pkl",
        "gradient_boosting_feature_importance.png",
    ),
}


for model_name, (model_path, output_path) in models.items():
    # load trained model
    model = joblib.load(model_path)

    # get feature names and importance
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()

    # Linear Regression uses coefficients instead of feature_importances_.
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    else:
        importances = abs(regressor.coef_)

    # create dataframe
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    })

    # Clean feature names
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

    # combine duplicate features
    feature_importance = (
        feature_importance
        .groupby("Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=True)
    )

    # results
    print(f"\n{model_name} Feature Importance:")
    print(feature_importance)

    # create chart
    plt.figure(figsize=(10, 7))

    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"EduPredict {model_name} Feature Importance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Graph Saved Successfully: {output_path}")
