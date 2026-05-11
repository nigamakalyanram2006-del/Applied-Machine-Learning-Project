#   ---Predicting student exam score using Linear Regression ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#   ---loading dataset---
df = pd.read_csv('enhanced_student_habits_performance_dataset.csv')
print(df)

#   ---checking for missing values---
print(df.isnull().sum())

#   ---feature engineering using mapping---
df = df.drop('student_id', axis=1)

# map categorical variables to numeric
df['part_time_job'] = df['part_time_job'].map({'No': 0, 'Yes': 1})
df['diet_quality'] = df['diet_quality'].map({'Poor': 0, 'Fair': 1, 'Good': 2})
df['parental_education_level'] = df['parental_education_level'].map({
    'High School': 0, 'Some College': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4
})
df['internet_quality'] = df['internet_quality'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['extracurricular_participation'] = df['extracurricular_participation'].map({'No': 0, 'Yes': 1})
df['dropout_risk'] = df['dropout_risk'].map({'No': 0, 'Yes': 1})
df['access_to_tutoring'] = df['access_to_tutoring'].map({'No': 0, 'Yes': 1})
df['family_income_range'] = df['family_income_range'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
df['major'] = df['major'].map({
    'Computer Science': 0, 'Arts': 1, 'Psychology': 2, 'Business': 3, 'Engineering': 4, 'Biology': 5
})
df['study_environment'] = df['study_environment'].map({
    'Co-Learning Group': 0, 'Library': 1, 'Quiet Room': 2, 'Dorm': 3, 'Cafe': 4
})
df['learning_style'] = df['learning_style'].map({
    'Reading': 0, 'Kinesthetic': 1, 'Visual': 2, 'Auditory': 3
})

print(df.head())
print(df.isnull().sum())

#   ---separating features and target---
X = df.drop('exam_score', axis=1)
y = df['exam_score']

#   ---creating testing and training sets---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#   ---standardization---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#   ---model initiation and training---
model = LinearRegression()
model.fit(X_train, y_train)

#   ---detecting overfitting/underfitting with cross validation---
train_score = model.score(X_train, y_train)
print(f"train score: {train_score}")

test_score = model.score(X_test, y_test)
print(f"test score: {test_score}")

cv_score = cross_val_score(model, X_train, y_train, cv=5)
print(f"cross validation score: {cv_score.mean()}")

#   ---prediction---
y_pred = model.predict(X_test)

#   ---model evaluation---
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'mean squared error: {mse}')
print(f'mean absolute error: {mae}')
print(f'r2 score: {r2}')

#   ---actual vs predicted---
plt.scatter(y_test, y_test, color='blue', s=5, label='Actual')
plt.scatter(y_test, y_pred, color='red', s=5, label='Predicted')
plt.title('Actual vs Predicted Exam Scores')
plt.legend()
plt.show()

#   ---new, unseen data to make predictions (from user)---
age = int(input('\nWhat is your Age\n >Your Answer: '))
gender = input('\nWhat is your gender\n >Male\n >Female\n >Other\n >Your Answer:  ')
major = input('\nWhat is your major\n >Computer Science\n >Arts\n >Psychology\n >Business\n >Engineering\n >Your Answer: ')
study_hours_per_day = float(input('\nWhat is your study hours (per day)\n >Your Answer: '))
social_media_hours = float(input('\nHow many hours do you spend on social media per day?\n >Your Answer: '))
netflix_hours = float(input('\nHow many hours do you spend watching Netflix/streaming per day?\n >Your Answer: '))
part_time_job = input('\nDo you have a part-time job?\n >Yes\n >No\n >Your Answer: ')
attendance_percentage = float(input('\nWhat is your attendance percentage?\n >Your Answer: '))
sleep_hours = float(input('\nHow many hours do you sleep per night?\n >Your Answer: '))
diet_quality = input('\nHow would you rate your diet quality?\n >Poor\n >Fair\n >Good\n >Your Answer: ')
exercise_frequency = int(input('\nHow many times do you exercise per week?\n >Your Answer: '))
parental_education_level = input(
    '\nWhat is your parental education level?\n'
    ' >High School\n >Some College\n >Bachelor\n >Master\n >PhD\n >Your Answer: '
)
internet_quality = input(
    '\nHow is your internet quality?\n'
    ' >Low\n >Medium\n >High\n >Your Answer: '
)
mental_health_rating = float(input('\nRate your mental health (1–10)\n >Your Answer: '))
extracurricular_participation = input(
    '\nDo you participate in extracurricular activities?\n'
    ' >Yes\n >No\n >Your Answer: '
)
previous_gpa = float(input('\nWhat is your previous GPA?\n >Your Answer: '))
semester = int(input('\nWhat semester are you in?\n >Your Answer: '))
stress_level = float(input('\nRate your stress level (1–10)\n >Your Answer: '))
dropout_risk = input('\nDo you consider yourself at risk of dropping out?\n >Yes\n >No\n >Your Answer: ')
social_activity = int(input('\nHow many social activities do you do per week?\n >Your Answer: '))
screen_time = float(input('\nWhat is your total screen time per day (hours)?\n >Your Answer: '))
study_environment = input(
    '\nWhere do you usually study?\n'
    ' >Co-Learning Group\n >Library\n >Quiet Room\n >Dorm\n >Cafe\n >Your Answer: '
)
access_to_tutoring = input('\nDo you have access to tutoring?\n >Yes\n >No\n >Your Answer: ')
family_income_range = input('\nWhat is your family income range?\n >Low\n >Medium\n >High\n >Your Answer: ')
parental_support_level = int(input('\nRate your parental support level (1–10)\n >Your Answer: '))
motivation_level = int(input('\nRate your motivation level (1–10)\n >Your Answer: '))
exam_anxiety_score = int(input('\nRate your exam anxiety (1–10)\n >Your Answer: '))
learning_style = input(
    '\nWhat is your learning style?\n'
    ' >Reading\n >Kinesthetic\n >Visual\n >Auditory\n >Your Answer: '
)
time_management_score = float(input('\nRate your time management skills (1–10)\n >Your Answer: '))

#   ---creating a dataframe from user input---
user_data = {
    "age": [age],
    "gender": [gender],
    "major": [major],
    "study_hours_per_day": [study_hours_per_day],
    "social_media_hours": [social_media_hours],
    "netflix_hours": [netflix_hours],
    "part_time_job": [part_time_job],
    "attendance_percentage": [attendance_percentage],
    "sleep_hours": [sleep_hours],
    "diet_quality": [diet_quality],
    "exercise_frequency": [exercise_frequency],
    "parental_education_level": [parental_education_level],
    "internet_quality": [internet_quality],
    "mental_health_rating": [mental_health_rating],
    "extracurricular_participation": [extracurricular_participation],
    "previous_gpa": [previous_gpa],
    "semester": [semester],
    "stress_level": [stress_level],
    "dropout_risk": [dropout_risk],
    "social_activity": [social_activity],
    "screen_time": [screen_time],
    "study_environment": [study_environment],
    "access_to_tutoring": [access_to_tutoring],
    "family_income_range": [family_income_range],
    "parental_support_level": [parental_support_level],
    "motivation_level": [motivation_level],
    "exam_anxiety_score": [exam_anxiety_score],
    "learning_style": [learning_style],
    "time_management_score": [time_management_score]
}
user_df = pd.DataFrame(user_data)

#   ---mapping user input like training data---
user_df['part_time_job'] = user_df['part_time_job'].map({'No': 0, 'Yes': 1})
user_df['diet_quality'] = user_df['diet_quality'].map({'Poor': 0, 'Fair': 1, 'Good': 2})
user_df['parental_education_level'] = user_df['parental_education_level'].map({
    'High School': 0, 'Some College': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4
})
user_df['internet_quality'] = user_df['internet_quality'].map({'Low': 0, 'Medium': 1, 'High': 2})
user_df['extracurricular_participation'] = user_df['extracurricular_participation'].map({'No': 0, 'Yes': 1})
user_df['dropout_risk'] = user_df['dropout_risk'].map({'No': 0, 'Yes': 1})
user_df['access_to_tutoring'] = user_df['access_to_tutoring'].map({'No': 0, 'Yes': 1})
user_df['family_income_range'] = user_df['family_income_range'].map({'Low': 0, 'Medium': 1, 'High': 2})
user_df['gender'] = user_df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
user_df['major'] = user_df['major'].map({
    'Computer Science': 0, 'Arts': 1, 'Psychology': 2, 'Business': 3, 'Engineering': 4, 'Biology': 5
})
user_df['study_environment'] = user_df['study_environment'].map({
    'Co-Learning Group': 0, 'Library': 1, 'Quiet Room': 2, 'Dorm': 3, 'Cafe': 4
})
user_df['learning_style'] = user_df['learning_style'].map({
    'Reading': 0, 'Kinesthetic': 1, 'Visual': 2, 'Auditory': 3
})

#   ---standardization---
user_df_scaled = scaler.transform(user_df)

#   ---prediction---
pred = model.predict(user_df_scaled)
print(f'\nPredicted Score: {pred[0]}%')