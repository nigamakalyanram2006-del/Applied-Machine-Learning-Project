#   ---Predicting student's G2 (midway) and G3 (final/at the very end of the year) scores using KNN ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#   ---loading dataset---
df = pd.read_csv('student-por.csv')

#   ---checking for missing values---
print(df.isnull().sum())

#   ---feature engineering---
pd.set_option('display.max_columns', None)
print(df)

print(df['school'].unique())
df['school'] = df['school'].map({'GP': 0, 'MS': 1})

print(df['sex'].unique())
df['sex'] = df['sex'].map({'F': 0, 'M': 1})

print(df['address'].unique())
df['address'] = df['address'].map({'U': 0, 'R': 1})

print(df['famsize'].unique())
df['famsize'] = df['famsize'].map({'LE3': 0, 'GT3': 1})

print(df['Pstatus'].unique())
df['Pstatus'] = df['Pstatus'].map({'A': 0, 'T': 1})

print(df['schoolsup'].unique())
df['schoolsup'] = df['schoolsup'].map({'no': 0, 'yes': 1})

print(df['famsup'].unique())
df['famsup'] = df['famsup'].map({'no': 0, 'yes': 1})

print(df['paid'].unique())
df['paid'] = df['paid'].map({'no': 0, 'yes': 1})

print(df['activities'].unique())
df['activities'] = df['activities'].map({'no': 0, 'yes': 1})

print(df['nursery'].unique())
df['nursery'] = df['nursery'].map({'no': 0, 'yes': 1})

print(df['higher'].unique())
df['higher'] = df['higher'].map({'no': 0, 'yes': 1})

print(df['internet'].unique())
df['internet'] = df['internet'].map({'no': 0, 'yes': 1})

print(df['romantic'].unique())
df['romantic'] = df['romantic'].map({'no': 0, 'yes': 1})

df = pd.get_dummies(df, columns=['Mjob', 'Fjob', 'reason', 'guardian'], drop_first=True)

print(df.isnull().sum())
print(df.isna().sum())
print(df)

#   ---separating features and targets---
corr = df.corr(numeric_only = True)
print(corr["G2"].sort_values(ascending=True))

X_g2 = df[['G1', 'failures', 'higher', 'school', 'Medu', 'studytime', 'Fedu']]
y_g2 = df['G2']

print(corr["G3"].sort_values(ascending=True))

X_g3 = df[['G1', 'G2', 'failures', 'higher', 'school', 'studytime', 'Medu']]
y_g3 = df['G3']

#   ---creating training and testing sets---
X_g2_train, X_g2_test, y_g2_train, y_g2_test = train_test_split(X_g2, y_g2, test_size = 0.2, random_state = 101)
X_g3_train, X_g3_test, y_g3_train, y_g3_test = train_test_split(X_g3, y_g3, test_size = 0.2, random_state = 101)

#   ---standardization---
scaler_g2 = StandardScaler()
X_g2_train = scaler_g2.fit_transform(X_g2_train)
X_g2_test = scaler_g2.transform(X_g2_test)

scaler_g3 = StandardScaler() 
X_g3_train = scaler_g3.fit_transform(X_g3_train)
X_g3_test = scaler_g3.transform(X_g3_test)

#   ---model initiation and training---
test_model_2 = KNeighborsRegressor()
params = [
    {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean"]
    },
    {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["manhattan"]
    },
    {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski"],
        "p": [1, 2]
    }
]
grid_2 = GridSearchCV(estimator=test_model_2, param_grid=params, cv=5, n_jobs = -1)
grid_2.fit(X_g2_train, y_g2_train)

print("Optimal Hyperparameters: ", grid_2.best_params_)
print("Best Score: ", grid_2.best_score_)

model_g2 = grid_2.best_estimator_
model_g2.fit(X_g2_train, y_g2_train)

train_score_g2 = model_g2.score(X_g2_train, y_g2_train)
print(f"train score g2: {train_score_g2}")

test_score_g2 = model_g2.score(X_g2_test, y_g2_test)
print(f"test score g2: {test_score_g2}")

cv_score_g2 = cross_val_score(model_g2, X_g2_train, y_g2_train, cv=5)
print(f"cross validation score g2: {cv_score_g2.mean()}")

test_model_3 = KNeighborsRegressor()
grid_3 = GridSearchCV(
    estimator=test_model_3,
    param_grid=params,
    cv=5,
    n_jobs=-1
)
grid_3.fit(X_g3_train, y_g3_train)

print("Optimal Hyperparameters: ", grid_3.best_params_)
print("Best Score: ", grid_3.best_score_)

model_g3 = grid_3.best_estimator_
model_g3.fit(X_g3_train, y_g3_train)

train_score_g3 = model_g3.score(X_g3_train, y_g3_train)
print(f"train score g3: {train_score_g3}")

test_score_g3 = model_g3.score(X_g3_test, y_g3_test)
print(f"test score g3: {test_score_g3}")

cv_score_g3 = cross_val_score(model_g3, X_g3_train, y_g3_train, cv=5)
print(f"cross validation score g3: {cv_score_g3.mean()}")

#   ---prediction---
y_g2_pred = model_g2.predict(X_g2_test)
y_g3_pred = model_g3.predict(X_g3_test)

#   ---model evaluation---
mse_g2 = mean_squared_error(y_g2_test, y_g2_pred)
mae_g2 = mean_absolute_error(y_g2_test, y_g2_pred)
r2_g2 = r2_score(y_g2_test, y_g2_pred)
print(f'mse: {mse_g2}, mae: {mae_g2}, r2 score: {r2_g2}')

mse_g3 = mean_squared_error(y_g3_test, y_g3_pred)
mae_g3 = mean_absolute_error(y_g3_test, y_g3_pred)
r2_g3 = r2_score(y_g3_test, y_g3_pred)
print(f'mse: {mse_g3}, mae: {mae_g3}, r2 score: {r2_g3}')

#   ---actual vs predicted (g2 and g2 grades only)---
plt.scatter(y_g2_test, y_g2_test, color = 'blue', label = 'Actual G2 Predictions')
plt.scatter(y_g2_test, y_g2_pred, color = 'red', label = "Predicted G2 Predictions")
plt.legend()
plt.show()

plt.scatter(y_g3_test, y_g3_test, color = 'blue', label = 'Actual G3 Predictions')
plt.scatter(y_g3_test, y_g3_pred, color = 'red', label = 'Predicted G3 Predictions')
plt.legend()
plt.show()

#   ---new unseen user data (g2 and g3 grades only)---

#   ---asking for inputs---
school = input("\nWhich school do you attend? \n >Type GP for Gabriel Pereira \n >Enter MS for Mousinho da Silveria \n>>>").upper()
higher = input("\nWill you be pursuing higher education (University)? \n >yes \n >no \n>>>").lower()
studytime = int(input("\nHow many hours do you study per week? \n>>>"))
Medu = int(input("\nWhat is your mother's education level? \n >0: none \n >1: primary (4th grade) \n >2: 5th-9th grade \n >3: secondary \n >4: higher \n>>>"))
Fedu = int(input("\nWhat is your father's education level? \n >0: none \n >1: primary (4th grade) \n >2: 5th-9th grade \n >3: secondary \n >4: higher \n>>>"))
failures = int(input("\nPast class failures (0-3, or 4 for more): \n>>>"))
g1 = int(input("\nWhat is your G1 (grade after first few months) Grade? \n>>>"))
g2 = int(input("\nWhat is your G2 (midpoint) Grade? (0-20) \n>>>"))

#   ---g2---
g2_data = {
    'G1' : [g1],
    'failures' : [failures],
    'higher' : [higher],
    'school' : [school],
    'Medu' : [Medu],
    'studytime' : [studytime],
    'Fedu' : [Fedu]            
}
g2_df = pd.DataFrame(g2_data)

#   ---feature engineering---
g2_df['school'] = g2_df['school'].map({'GP': 0, 'MS': 1})
g2_df['higher'] = g2_df['higher'].map({'no': 0, 'yes': 1})

#   ---standardization---
g2_df = scaler_g2.transform(g2_df)

#   ---prediction---
pred_g2 = model_g2.predict(g2_df)
print(f'Your predicted G2 (midway) grade is {pred_g2[0]}')

#   ---g3---
g3_data = {
    'G1' : [g1],
    'G2' : [g2],
    'failures' : [failures],
    'higher' : [higher],
    'school' : [school],
    'studytime' : [studytime],
    'Medu' : [Medu]            
}
g3_df = pd.DataFrame(g3_data)

#   ---feature engineering---
g3_df['school'] = g3_df['school'].map({'GP': 0, 'MS': 1})
g3_df['higher'] = g3_df['higher'].map({'no': 0, 'yes': 1})

#   ---standardization---
g3_df = scaler_g3.transform(g3_df)

#   ---prediction---
pred_g3 = model_g3.predict(g3_df)
print(f'Your predicted G3 (final/end of the year) grade is {pred_g3[0]}')

#   ---grading scale reference---
print("\n--- PORTUGUESE TO US GRADE REFERENCE ---")
print("18 – 20: A+ / A  | Excellent / Outstanding")
print("16 – 17: A- / B+ | Very Good")
print("14 – 15: B / B-  | Good")
print("10 – 13: C / D   | Sufficient (Passing)")
print(" 0 –  9: F       | Fail")
print("-" * 40)
