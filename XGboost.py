#   ---Predicting LLM Rank using XGBoost Regressor and predicting LLM model type, using Classifer ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, jaccard_score, classification_report, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier

#   ---loading dataset---
df = pd.read_csv("llm_benchmark_comparison_2025_2026.csv")

#   ---feature engineering---
le = LabelEncoder()
df['organization'] = le.fit_transform(df['organization'])
df['license'] = le.fit_transform(df['license'])
df['best_for'] = le.fit_transform(df['best_for'])

df['speed_tier'] = df['speed_tier'].map({'Slow': 0, 'Medium': 1, 'Fast': 2, 'Ultra-Fast': 3})
df['price_tier'] = df['price_tier'].map({'Free/Ultra-cheap': 0, 'Budget': 1, 'Mid-range': 2, 'Premium': 3})
df['context_tier'] = df['context_tier'].map({'Standard': 0, 'Large': 1, 'Extended': 2, 'Massive': 3})

df['open_source'] = df['open_source'].map({True: 1, False: 0})
df['reasoning_model'] = df['reasoning_model'].map({True: 1, False: 0})
df['multimodal'] = df['multimodal'].map({True: 1, False: 0})

df = pd.get_dummies(df, columns=['country', 'architecture', 'modality', 'type'])

df = df.drop(columns=['model_name'])

pd.set_option('display.max_columns', None)
print(df.isnull().sum())
print(df.isna().sum())

#   ---separating features and target---
X_reg = df[['mmlu', 'gpqa_diamond', 'humaneval', 'parameters_b', 'input_price_per_1m', 'type_Proprietary']]
y_reg = df['rank']

X_class = df[['price_tier', 'performance_per_dollar', 'gpqa_diamond', 'architecture_MoE','parameters_b', 'context_window_k']]

mapping = {'type_Proprietary': 0, 'type_Open Source': 1, 'type_Open Weight': 2}
df['type_target'] = df[['type_Open Source', 'type_Open Weight', 'type_Proprietary']].idxmax(axis=1).map(mapping)

corr = df.corr(numeric_only=True)
print(corr['type_target'].sort_values(ascending=False))
y_class = df['type_target']

#   ---creating training and testing sets---
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size = 0.2, random_state = 101)
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size = 0.2, random_state = 101, stratify=y_class)

#   ---model initiation and training---
model_reg = XGBRegressor(random_state = 101)
params = {"n_estimators":[100, 200, 300, 400, 500], "learning_rate":[0.01, 0.05, 0.1], "max_depth":[3, 5, 7], "subsample": [0.7, 0.8, 1.0], "colsample_bytree": [0.7, 0.8, 1.0], "min_child_weight": [1, 3, 5]}
grid = GridSearchCV(estimator = model_reg, param_grid = params, cv = 5, n_jobs = -1)
grid.fit(X_reg_train, y_reg_train)

print("Optimal Hyperparameters: ",grid.best_params_)
print("Best Score: ", grid.best_score_)

xgbreg = grid.best_estimator_
xgbreg.fit(X_reg_train, y_reg_train)

#   ---detecting overfitting/underfitting with cross validation---
train_score_reg = xgbreg.score(X_reg_train, y_reg_train)
print(f"train score reg: {train_score_reg}")

test_score_reg = xgbreg.score(X_reg_test, y_reg_test)
print(f"test score reg: {test_score_reg}")

cv_score_reg = cross_val_score(xgbreg, X_reg_train, y_reg_train, cv=5)
print(f"cross validation score reg: {cv_score_reg.mean()}")

model_class = XGBClassifier(random_state = 101)
params = {
    "n_estimators": [50, 100],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1]
}
grid = GridSearchCV(estimator=model_class, param_grid=params, cv = 5, n_jobs = -1)
grid.fit(X_class_train, y_class_train)

print("Optimal Hyperparameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

xgbclass = grid.best_estimator_
xgbclass.fit(X_class_train, y_class_train)

train_score_class = xgbclass.score(X_class_train, y_class_train)
print(f"train score class: {train_score_class}")

test_score_class = xgbclass.score(X_class_test, y_class_test)
print(f"test score class: {test_score_class}")

cv_score_class = cross_val_score(xgbclass, X_class_train, y_class_train, cv=3)
print(f"cross validation score class: {cv_score_class.mean()}")

#   ---prediction---
y_reg_pred = xgbreg.predict(X_reg_test)
y_class_pred = xgbclass.predict(X_class_test)

#   ---model evaluation---
mse = mean_squared_error(y_reg_test, y_reg_pred)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)
print(f"mse: {mse}, mae: {mae}, r2 score: {r2}")

cr = classification_report(y_class_test, y_class_pred, zero_division=0)
print(cr)
js = jaccard_score(y_class_test, y_class_pred, average='weighted')
print("Jaccard Score: ",js)
cm = confusion_matrix(y_class_test, y_class_pred)
print(cm)

#   ---new, unseen llm data---
m_mmlu = float(input("\nEnter MMLU score: "))
m_gpqa = float(input("Enter GPQA Diamond score: "))
m_heval = float(input("Enter HumanEval score: "))
m_params = float(input("Enter Parameters (B): "))
m_price_token = float(input("Enter Input Price per 1M: "))
m_is_prop = int(input("Is it Proprietary? (1=Yes, 0=No): "))
m_tier = int(input("Enter Price Tier (0-3): "))
m_per_dollar = float(input("Enter Performance per Dollar: "))
m_moe = int(input("Is it MoE? (1=Yes, 0=No): "))
m_context = float(input("Enter Context Window (K): "))

df_reg = pd.DataFrame([{
    'mmlu': m_mmlu,
    'gpqa_diamond': m_gpqa,
    'humaneval': m_heval,
    'parameters_b': m_params,
    'input_price_per_1m': m_price_token,
    'type_Proprietary': m_is_prop
}])

df_class = pd.DataFrame([{
    'price_tier': m_tier,
    'performance_per_dollar': m_per_dollar,
    'gpqa_diamond': m_gpqa,
    'architecture_MoE': m_moe,
    'parameters_b': m_params,
    'context_window_k': m_context
}])

reg_pred = xgbreg.predict(df_reg)
print(f"\nYour LLM's Rank is: {reg_pred[0]}")

class_pred = xgbclass.predict(df_class)

prob_prop = xgbclass.predict_proba(df_class)[:,0]
prob_opensource = xgbclass.predict_proba(df_class)[:,1]
prob_openweight = xgbclass.predict_proba(df_class)[:,2]

per_prop = prob_prop * 100
per_opensource = prob_opensource * 100
per_openweight = prob_openweight * 100

if class_pred[0] == 0:
    print(f"This model is predicted to be PROPRIETARY\nConfidence Level: {per_prop}%")
elif class_pred[0] == 1:
    print(f"This model is predicted to be OPEN SOURCE\nConfidence Level: {per_opensource}%")
elif class_pred[0] == 2:
    print(f"This model is predicted to be OPEN WEIGHT\nConfidence Level: {per_openweight}%")