import pandas as pd

def preprocess_data(df):
    processed_df = df.copy()

    # -----------------------------------------------
    # Ordinal
    # ===============================================

    ordinal_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    ordinal_columns = [
        "Parental_Involvement",
        "Access_to_Resources",
        "Motivation_Level",
        "Family_Income",
        "Teacher_Quality"
    ]

    for column in ordinal_columns:
        processed_df[column] = processed_df[column].map(ordinal_mapping)

    # Ordinal: Parental Education
    # ===============================================

    education_mapping = {
        "High School": 0,
        "College": 1,
        "Postgraduate": 2
    }

    processed_df["Parental_Education_Level"] = (
        processed_df["Parental_Education_Level"]
        .map(education_mapping)
    )

    # Ordinal: Distance from home
    # ===============================================

    distance_mapping = {
        "Near": 0,
        "Moderate": 1,
        "Far": 2
    }

    processed_df["Distance_from_Home"] = (
        processed_df["Distance_from_Home"]
        .map(distance_mapping)
    )


    # -----------------------------------------------
    # Binary
    # ===============================================

    binary_mapping = {
        "No": 0,
        "Yes": 1
    }

    binary_columns = [
        "Extracurricular_Activities",
        "Internet_Access",
        "Learning_Disabilities"
    ]

    for column in binary_columns:
        processed_df[column] = processed_df[column].map(binary_mapping)

    # Nominal: One hot encoding
    # ===============================================

    nominal_columns = [
        "School_Type",
        "Peer_Influence",
        "Gender"
    ]

    processed_df = pd.get_dummies(
        processed_df,
        columns=nominal_columns,
        dtype=int
    )

    return processed_df