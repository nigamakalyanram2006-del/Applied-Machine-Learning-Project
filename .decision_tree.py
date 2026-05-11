#  ---Predicting Diamond Price and Cut using Decision Tree ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

#   ---loading dataset---
df = pd.read_csv('diamonds.csv')
print(df)

#   ---checking for missing values---
print(df.isnull().sum())

#   ---feature engineering---
df = df.drop('Unnamed: 0', axis = 1)

pd.set_option('display.max_columns', None)
print(df.head())

le = LabelEncoder()
df['cut'] = le.fit_transform(df['cut'])

df['color'] = df['color'].map({
    'D':0, 'E':1, 'F':2, 'G':3, 'H':4, 'I':5, 'J':6
})

print(df['clarity'].unique())
df['clarity'] = df['clarity'].map({'I1':0, 'SI2':1, 'SI1':2, 'VS2':3, 'VS1':4, 'VVS2':5, 'VVS1':6, 'IF':7})

print(df)
print(df.isnull().sum())
print(df.isna().sum())

#   ---separating features and target---
X_cut = df.drop('cut', axis = 1)
y_cut = df['cut']

X_price = df.drop('price', axis = 1)
y_price = df['price']

#   ---creating training and testing sets---
X_cut_train, X_cut_test, y_cut_train, y_cut_test = train_test_split(X_cut, y_cut, test_size = 0.2, random_state = 101)
X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size = 0.2, random_state = 101)

#   ---using cross validation to hypertune parameters---
for depth in range(1,21 ):
    model = DecisionTreeClassifier(max_depth = depth, random_state = 101)
    scores = cross_val_score(model, X_cut_train, y_cut_train, cv = 5)
    print(f"max depth: {depth}, cv score: {scores.mean()}")

#   ---model initiation---
model_class = DecisionTreeClassifier(max_depth = 12, random_state = 101)
model_class.fit(X_cut_train, y_cut_train)

#   ---detecting overfitting/underfitting with cross validation---
train_score_class = model_class.score(X_cut_train, y_cut_train)
print(f"train score class: {train_score_class}")

test_score_class = model_class.score(X_cut_test, y_cut_test)
print(f"test score class: {test_score_class}")

cv_score_class = cross_val_score(model_class, X_cut_train, y_cut_train, cv=5)
print(f"cross validation score class: {cv_score_class.mean()}")

model_reg = DecisionTreeRegressor()
model_reg.fit(X_price_train, y_price_train)

train_score_reg = model_reg.score(X_price_train, y_price_train)
print(f"train score reg: {train_score_reg}")

test_score_reg = model_reg.score(X_price_test, y_price_test)
print(f"test score reg: {test_score_reg}")

cv_score_reg = cross_val_score(model_reg, X_price_train, y_price_train, cv=5)
print(f"cross validation score reg: {cv_score_reg.mean()}")


#   ---prediction---
y_cut_pred = model_class.predict(X_cut_test)
y_price_pred = model_reg.predict(X_price_test)

#   ---model evaluation---
cr = classification_report(y_cut_test, y_cut_pred)
cm = confusion_matrix(y_cut_test, y_cut_pred)
print(cr)
print(cm)

mse = mean_squared_error(y_price_test, y_price_pred)
mae = mean_absolute_error(y_price_test, y_price_pred)
r2 = r2_score(y_price_test, y_price_pred)
print(f'mse: {mse}, mae: {mae}, r2 score: {r2}')

#   ---new unseen user data---
carat = float(input("\nHow heavy is the diamond in carats? (e.g. 0.5, 1.2) \n>>>"))
cut = int(input("\nWhat is your Diamond's cut? \n>>>"))
color = int(input("\nWhat is the color grade? \n >Enter 0 for E \n >Enter 1 for I \n >Enter 2 for J \n >Enter 3 for H \n >Enter 4 for F \n >Enter 5 for G \n >Enter 6 for D \n>>> "))
clarity = int(input("\nHow clear is the diamond? \n >Enter 0 for I1 \n >Enter 1 for SI2 \n >Enter 2 for SI1 \n >Enter 3 for VS2 \n >Enter 4 for VS1 \n >Enter 5 for VVS2 \n >Enter 6 for VVS1 \n >Enter 7 for IF \n>>> "))
depth = float(input("\nWhat is the total depth percentage? (Usually between 43-79) \n>>> "))
table = float(input("\nWhat is the table width percentage? (Usually between 43-95) \n>>> "))
price = int(input("\nEnter the s the current market price of the Diamond? \n>>>"))
x = float(input("\nEnter the Length (x): \n>>> "))
y = float(input("\nEnter the Width (y): \n>>> "))
z = float(input("\nEnter the Depth (z): \n>>> "))

cut_data = {
    'carat':[carat],
    'color':[color],
    'clarity':[clarity],
    'depth':[depth],
    'table':[table],
    'x':[x],
    'y':[y],
    'z':[z]
}
cut_df = pd.DataFrame(cut_data)

price_data = {
    'carat' : [carat],
    'color' : [color],
    'clarity' : [clarity],
    'depth' : [depth],
    'table' : [table],
    'x' : [x],
    'y' : [y],
    'z' : [z]
}
price_df = pd.DataFrame(price_data)

#   ---predictions---
pred_cut = model_class.predict(cut_df)
pred_price = model_reg.predict(price_df)

print(dict(zip(le.classes_, le.transform(le.classes_))))

if pred_cut[0] == 0:
    print(f'\nYour predicted Diamond Cut is: Fair')
elif pred_cut[0] == 1:
    print(f'\nYour predicted Diamond Cut is: Good')
elif pred_cut[0] == 2:
    print(f'\nYour predicted Diamond Cut is: Ideal')
elif pred_cut[0] == 3:
    print(f'\nYour predicted Diamond Cut is: Premium')
elif pred_cut[0] == 4:
    print(f'\nYour predicted Diamond Cut is: Very Good')

print(f'\nYour predicted Diamond Price is {pred_price[0]}')