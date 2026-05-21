#   ---Predicting House Prices (Advanced Regression Techniques) using Ridge ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#   ---loading dataset---
df = pd.read_csv("train.csv")

#   ---feature engineering---
pd.set_option('display.max_columns', None)
print(df.head())

df.drop(columns=['Id'], inplace=True)

df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

df['ExterQual'] = df['ExterQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['ExterCond'] = df['ExterCond'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['HeatingQC'] = df['HeatingQC'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
df['KitchenQual'] = df['KitchenQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

le = LabelEncoder()
binary_cols = ['Street', 'CentralAir']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, drop_first=True)

print(df.isnull().sum())

#   ---separating features and target---
X = df.drop('SalePrice', axis = 1)
y = np.log1p(df['SalePrice'])

#   ---creating training and testing sets---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=101)

#   ---standardization---
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#   ---gridsearchcv for optimal hyperparameters---
test_model = Ridge(max_iter = 10000)

params = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 600, 700, 800, 900, 1000]}

grid = GridSearchCV(estimator=test_model, param_grid=params, cv = 5, n_jobs = -1)
grid.fit(X_train, y_train)

print("Best optimal hyperparameters: ",grid.best_params_)
print("Best Score: ", grid.best_score_)

#   ---model initiation and training---
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
new_house = pd.DataFrame(columns=X.columns)
new_house.loc[0] = 0

new_house['MSSubClass'] = 60
new_house['LotFrontage'] = 70.0
new_house['LotArea'] = 9000
new_house['OverallQual'] = 8
new_house['OverallCond'] = 5
new_house['YearBuilt'] = 2005
new_house['YearRemodAdd'] = 2005
new_house['MasVnrArea'] = 150.0
new_house['ExterQual'] = 4   
new_house['ExterCond'] = 3  
new_house['HeatingQC'] = 5   
new_house['KitchenQual'] = 4 
new_house['BsmtFinSF1'] = 600
new_house['BsmtFinSF2'] = 0
new_house['BsmtUnfSF'] = 200
new_house['TotalBsmtSF'] = 800
new_house['1stFlrSF'] = 900
new_house['2ndFlrSF'] = 800
new_house['LowQualFinSF'] = 0
new_house['GrLivArea'] = 1700
new_house['BsmtFullBath'] = 1
new_house['BsmtHalfBath'] = 0
new_house['FullBath'] = 2
new_house['HalfBath'] = 1
new_house['BedroomAbvGr'] = 3
new_house['KitchenAbvGr'] = 1
new_house['TotRmsAbvGrd'] = 8
new_house['Fireplaces'] = 1
new_house['GarageYrBlt'] = 2005
new_house['GarageCars'] = 2
new_house['GarageArea'] = 500
new_house['WoodDeckSF'] = 100
new_house['OpenPorchSF'] = 40
new_house['EnclosedPorch'] = 0
new_house['3SsnPorch'] = 0
new_house['ScreenPorch'] = 0
new_house['PoolArea'] = 0
new_house['MiscVal'] = 0
new_house['MoSold'] = 6
new_house['YrSold'] = 2008
new_house['Street'] = 1   
new_house['CentralAir'] = 1  # Y


new_house['MSZoning_RL'] = 1
new_house['Neighborhood_CollgCr'] = 1
new_house['Condition1_Norm'] = 1
new_house['Condition2_Norm'] = 1
new_house['HouseStyle_2Story'] = 1
new_house['RoofStyle_Gable'] = 1
new_house['RoofMatl_CompShg'] = 1
new_house['Exterior1st_VinylSd'] = 1
new_house['Exterior2nd_VinylSd'] = 1
new_house['MasVnrType_BrkFace'] = 1
new_house['Foundation_PConc'] = 1
new_house['BsmtQual_Gd'] = 1
new_house['BsmtCond_TA'] = 1
new_house['BsmtExposure_No'] = 1
new_house['BsmtFinType1_GLQ'] = 1
new_house['BsmtFinType2_Unf'] = 1
new_house['Heating_GasA'] = 1
new_house['Electrical_SBrkr'] = 1
new_house['Functional_Typ'] = 1
new_house['FireplaceQu_TA'] = 1
new_house['GarageType_Attchd'] = 1
new_house['GarageFinish_RFn'] = 1
new_house['GarageQual_TA'] = 1
new_house['GarageCond_TA'] = 1
new_house['PavedDrive_Y'] = 1
new_house['SaleType_WD'] = 1
new_house['SaleCondition_Normal'] = 1

new_house_scaled = scaler.transform(new_house)
log_pred = model.predict(new_house_scaled)
predicted_price = np.expm1(log_pred[0])

print(f"\nThe predicted house price is: ${predicted_price}")