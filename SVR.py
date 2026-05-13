#   ---Predicting car selling price using SVR ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#   ---loading dataset---
df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
print(df)

#   ---feature engineering---
df['brand'] = df['name'].apply(lambda x: x.split(' ')[0])
df = df.drop('name', axis=1)

print(df['fuel'].unique()) 
print(df['seller_type'].unique())  
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'brand'], drop_first = True)

print(df['transmission'].unique())  
df['transmission'] = df['transmission'].map({"Manual":0, "Automatic":1})

print(df['owner'].unique())  
df['owner'] = df['owner'].map({"Test Drive Car":0, "First Owner":1, "Second Owner":2, "Third Owner":3, "Fourth & Above Owner":4})

print(df)

#   ---seprating features and target(s)---
X = df.drop('selling_price', axis = 1)
y = np.log1p(df['selling_price'])

#   ---creating training and testing sets---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

#   ---standardization---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#   ---gridsearchcv for optimal hyperparameters---
test_model = SVR()

params = {"C":[0.1, 1, 10, 100], "gamma":[1, 0.1, 0.01, 0.001], "kernel":['rbf', 'linear']}

grid = GridSearchCV(estimator=test_model, param_grid=params, cv=5, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

print("Optimal Hyperparameters: ", grid.best_params_)
print("Best Score: ", grid.best_score_)


#   ---model initiation and traning---
model = grid.best_estimator_
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
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"mae: {mae}, mse: {mse}, r2 score: {r2}")

#   ---comparing actual vs predicted---
plt.scatter(y_test, y_test, color = "red", label = "Actual Values")
plt.scatter(y_test, y_pred, color = "blue", label = "Predicted Values")
plt.legend()
plt.show()

#   ---new, unseen user data---
user_data = [
    {
        "name": "Toyota Fortuner 2.8",
        "year": 2021,
        "km_driven": 15000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Automatic",
        "owner": "First Owner"
    },
    {
        "name": "Honda City i-VTEC V",
        "year": 2015,
        "km_driven": 65000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "Second Owner"
    },
    {
        "name": "Maruti Swift VDI",
        "year": 2012,
        "km_driven": 120000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "Third Owner"
    },
    {
        "name": "BMW 5 Series 520d",
        "year": 2018,
        "km_driven": 35000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Automatic",
        "owner": "First Owner"
    },
    {
        "name": "Ford EcoSport 1.5",
        "year": 2019,
        "km_driven": 28000,
        "fuel": "Petrol",
        "seller_type": "Trustmark Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
    }
]
user_df = pd.DataFrame(user_data)

user_df['brand'] = user_df['name'].apply(lambda x: x.split(' ')[0])
user_df = user_df.drop('name', axis=1)

user_df['transmission'] = user_df['transmission'].map({"Manual": 0, "Automatic": 1})
user_df['owner'] = user_df['owner'].map({
    "Test Drive Car": 0, 
    "First Owner": 1, 
    "Second Owner": 2, 
    "Third Owner": 3, 
    "Fourth & Above Owner": 4
})

user_df = pd.get_dummies(user_df, columns=['fuel', 'seller_type', 'brand'], drop_first=True)

user_df = user_df.reindex(columns=X.columns, fill_value=0)

user_df = scaler.transform(user_df)
user_log_preds = model.predict(user_df)

final_prices = np.expm1(user_log_preds)

for i in range(len(user_data)):
    print(user_data[i]['name'], ":", final_prices[i])
