import joblib
import pandas as pd

#loading trained model
model = joblib.load("models/student_performance_model.pkl")

#new student
student = pd.DataFrame([{
    "Hours_Studied": 25,
    "Attendance": 90,
    "Parental_Involvement": "High",
    "Access_to_Resources": "High",
    "Extracurricular_Activities": "Yes",
    "Sleep_Hours": 7,
    "Previous_Scores": 75,
    "Motivation_Level": "High",
    "Internet_Access": "Yes",
    "Tutoring_Sessions": 2,
    "Family_Income": "Medium",
    "Teacher_Quality": "High",
    "Physical_Activity": 4,
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "College",
    "Distance_from_Home": "Near",
    "School_Type": "Public",
    "Peer_Influence": "Positive",
    "Gender": "Male"
}])

#Make Preediction
prediction = model.predict(student)

print("Predicted Exam Score:", prediction[0])