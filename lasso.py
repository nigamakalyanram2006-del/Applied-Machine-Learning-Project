#   ---Predicing the amount of thermal energy that a building's heating and air systems add or remove to keep building indoor temp comfortable based on the building's design, using Lasso Regression ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#   ---loading dataset---
df = pd.read_csv("energy_efficiency_data.csv")
print(df.isnull().sum())

#   ---feature engineering---
pd.set_option('display.max_columns', None)
print(df.head())

#   ---separating feautures and targets(s)---
X_heat = df.drop(['Heating_Load', 'Cooling_Load'], axis = 1)
y_heat = df['Heating_Load']

X_cool = df.drop(['Heating_Load', 'Cooling_Load'], axis = 1)
y_cool = df['Cooling_Load']

#   ---creating training and testing sets---
X_heat_train, X_heat_test, y_heat_train, y_heat_test = train_test_split(X_heat, y_heat, test_size = 0.2, random_state = 101)
X_cool_train, X_cool_test, y_cool_train, y_cool_test = train_test_split(X_cool, y_cool, test_size = 0.2, random_state = 101)

#   ---standardization---
scaler_heat = StandardScaler()
scaler_cool = StandardScaler()

X_heat_train = scaler_heat.fit_transform(X_heat_train)
X_heat_test = scaler_heat.transform(X_heat_test)
X_cool_train = scaler_cool.fit_transform(X_cool_train)
X_cool_test = scaler_cool.transform(X_cool_test)

#   ---gridsearchcv for optimal hyperparameters---
test_heat = Lasso(max_iter=10000)

heat_params = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

heat_grid = GridSearchCV(estimator=test_heat, param_grid=heat_params, cv = 5, n_jobs = -1)
heat_grid.fit(X_heat_train, y_heat_train)

print("Best optimal Heat hyperparameters: ",heat_grid.best_params_)
print("Best Heat Score: ", heat_grid.best_score_)

test_cool = Lasso(max_iter=10000)

cool_params = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

cool_grid = GridSearchCV(estimator=test_cool, param_grid = cool_params, cv = 5, n_jobs = -1)
cool_grid.fit(X_cool_train, y_cool_train)

print("Best optimal Cool hyperparameters: ", cool_grid.best_params_)
print("Best Cool SCore: ", cool_grid.best_score_)

#   ---model initiation and training---
model_heat = heat_grid.best_estimator_
model_heat.fit(X_heat_train, y_heat_train)

model_cool = cool_grid.best_estimator_
model_cool.fit(X_cool_train, y_cool_train)

#   ---detecting overfitting/underfitting with cross validation---
train_heat_score = model_heat.score(X_heat_train, y_heat_train)
print(f"heat train score: {train_heat_score}")

test_heat_score = model_heat.score(X_heat_test, y_heat_test)
print(f"heat test score: {test_heat_score}")

cv_heat_score = cross_val_score(model_heat, X_heat_train, y_heat_train, cv=5)
print(f"heat cross validation score: {cv_heat_score.mean()}")

train_cool_score = model_cool.score(X_cool_train, y_cool_train)
print(f"cool train score: {train_cool_score}")

test_cool_score = model_cool.score(X_cool_test, y_cool_test)
print(f"cool test score: {test_cool_score}")

cv_cool_score = cross_val_score(model_cool, X_cool_train, y_cool_train, cv=5)
print(f"cool cross validation score: {cv_cool_score.mean()}")

#   ---prediction---
y_heat_pred = model_heat.predict(X_heat_test)
y_cool_pred = model_cool.predict(X_cool_test)

#   ---model evaluation---
heat_mae = mean_absolute_error(y_heat_test, y_heat_pred)
heat_mse = mean_squared_error(y_heat_test, y_heat_pred)
heat_r2 = r2_score(y_heat_test, y_heat_pred)
print(f"Heat:\nmae: {heat_mae}, mse: {heat_mse}, r2: {heat_r2}")

cool_mae = mean_absolute_error(y_cool_test, y_cool_pred)
cool_mse = mean_squared_error(y_cool_test, y_cool_pred)
cool_r2 = r2_score(y_cool_test, y_cool_pred)
print(f"Cool:\nmae: {cool_mae}, mse: {cool_mse}, r2: {cool_r2}")

#   ---new, unseen user data---
Relative_Compactness = float(input("\nRelative_Compactness = "))
Surface_Area = float(input("Surface_Area = "))
Wall_Area = float(input("Wall_Area = "))
Roof_Area = float(input("Roof_Area = "))
Overall_Height = float(input("Overall_Height = "))
Orientation = float(input("Orientation (2: North, 3: East, 4: South, 5: West) = "))
Glazing_Area = float(input("Glazing_Area = "))
Glazing_Area_Distribution = float(input("Glazing_Area_Distribution = "))

user_df = pd.DataFrame(
    [
        [
            Relative_Compactness,
            Surface_Area,
            Wall_Area,
            Roof_Area,
            Overall_Height,
            Orientation,
            Glazing_Area,
            Glazing_Area_Distribution,
        ]
    ],
    columns=[
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Orientation",
        "Glazing_Area",
        "Glazing_Area_Distribution",
    ],
)

user_heat_df = scaler_heat.transform(user_df)
user_cool_df = scaler_cool.transform(user_df)

user_heat_pred = model_heat.predict(user_heat_df)
user_cool_pred = model_cool.predict(user_cool_df)

print(f"\nPredicted Heating Load: ", user_heat_pred[0])
print(f"Predicted Cooling Load: ", user_cool_pred[0])