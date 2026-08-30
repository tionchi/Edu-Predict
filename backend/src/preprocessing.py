from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def create_preprocessor():

    # -----------------------------------------------
    # Numeric Columns
    # ===============================================

    numeric_columns = [
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Scores",
        "Tutoring_Sessions",
        "Physical_Activity"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # -----------------------------------------------
    # Ordinal Columns
    # ===============================================

    ordinal_columns = [
        "Parental_Involvement",
        "Access_to_Resources",
        "Motivation_Level",
        "Family_Income",
        "Teacher_Quality",
        "Parental_Education_Level",
        "Distance_from_Home"
    ]

    ordinal_categories = [
        ["Low", "Medium", "High"],             # Parental_Involvement
        ["Low", "Medium", "High"],             # Access_to_Resources
        ["Low", "Medium", "High"],             # Motivation_Level
        ["Low", "Medium", "High"],             # Family_Income
        ["Low", "Medium", "High"],             # Teacher_Quality
        ["High School", "College", "Postgraduate"],  # Education
        ["Near", "Moderate", "Far"]             # Distance
    ]

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ordinal_categories
        ))
    ])

    # -----------------------------------------------
    # Binary
    # ===============================================

    binary_columns = [
        "Extracurricular_Activities",
        "Internet_Access",
        "Learning_Disabilities"
    ]

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="if_binary",
            handle_unknown="ignore"
        ))
    ])

    # -----------------------------------------------
    # Nominal
    # ===============================================

    nominal_columns = [
        "School_Type",
        "Peer_Influence",
        "Gender"
    ]

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    # -----------------------------------------------
    # Combination
    # ===============================================

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("ordinal", ordinal_pipeline, ordinal_columns),
        ("binary", binary_pipeline, binary_columns),
        ("nominal", nominal_pipeline, nominal_columns)
    ])

    return preprocessor
