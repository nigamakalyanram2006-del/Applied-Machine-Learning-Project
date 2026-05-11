#   ---Predicting churning customers and their credit card limit using Random Forest ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve

#   ---loading dataset---
df = pd.read_csv('BankChurners.csv')

df = df.drop('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', axis = 1)
df = df.drop('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2', axis = 1)
print(df)

#   ---checking for missing values---
print(df.isnull().sum())

#   ---feature engineering---
pd.set_option('display.max_columns', None)
print(df.head())

df = df.drop('CLIENTNUM', axis = 1)

print(df['Attrition_Flag'].unique())
df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

print(df['Gender'].unique())
df['Gender'] = df['Gender'].map({'M':0,'F':1})

print(df['Education_Level'].unique())
df['Education_Level'] = df['Education_Level'].map({
    'Uneducated': 0,
    'High School': 1,
    'College': 2,
    'Graduate': 3,
    'Post-Graduate': 4,
    'Doctorate': 5,
    'Unknown': -1   
})

print(df['Income_Category'].unique())
df['Income_Category'] = df['Income_Category'].map({
    'Less than $40K': 0,
    '$40K - $60K': 1,
    '$60K - $80K': 2,
    '$80K - $120K': 3,
    '$120K +': 4,
    'Unknown': -1
})

print(df['Card_Category'].unique())
df = pd.get_dummies(df, columns=['Marital_Status', 'Card_Category'], 
                    prefix=['Marital_Status', 'Card_Category'])

print(df)
print(df.isnull().sum())
print(df.isna().sum())

#   ---separating features and target---
#classification
X_chur = df.drop(['Attrition_Flag', 'Credit_Limit'], axis = 1)
y_chur = df['Attrition_Flag']
#regression
X_card = df.drop([
    'Credit_Limit',
    'Attrition_Flag',
    'Avg_Open_To_Buy'
], axis=1)
y_card = df['Credit_Limit']

#   ---creating training and testing sets---
X_chur_train, X_chur_test, y_chur_train, y_chur_test = train_test_split(X_chur, y_chur, test_size = 0.2, random_state=101)
X_card_train, X_card_test, y_card_train, y_card_test = train_test_split(X_card, y_card, test_size = 0.2, random_state=101)

#   ---model initiation and training---
model_chur = RandomForestClassifier(n_estimators=200, max_depth = 10, min_samples_split = 20, min_samples_leaf = 10, random_state=101)
model_chur.fit(X_chur_train, y_chur_train)

#   ---detecting overfitting/underfitting with cross validation---
train_score_chur = model_chur.score(X_chur_train, y_chur_train)
print(f"train score chur: {train_score_chur}")

test_score_chur = model_chur.score(X_chur_test, y_chur_test)
print(f"test score chur: {test_score_chur}")

cv_score_chur = cross_val_score(model_chur, X_chur_train, y_chur_train, cv=5)
print(f"cross validation score chur: {cv_score_chur.mean()}")

model_card = RandomForestRegressor(n_estimators=100, random_state=101)
model_card.fit(X_card_train, y_card_train)

train_score_card = model_card.score(X_card_train, y_card_train)
print(f"train score card: {train_score_card}")

test_score_card = model_card.score(X_card_test, y_card_test)
print(f"test score card: {test_score_card}")

cv_score_card = cross_val_score(model_card, X_card_train, y_card_train, cv=5)
print(f"cross validation score card: {cv_score_card.mean()}")

#   ---prediction---
y_chur_pred = model_chur.predict(X_chur_test)
y_card_pred = model_card.predict(X_card_test)

#   ---model evaluation---
cr = classification_report(y_chur_test, y_chur_pred)
cm = confusion_matrix(y_chur_test, y_chur_pred)
print(f'classification report: {cr}')
print(f'confusion matrix: {cm}')

cmdisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Existing Customer','Attrited Customer'])
cmdisp.plot()

y_prob = model_chur.predict_proba(X_chur_test)[:,1]
auc = roc_auc_score(y_chur_test, y_prob)
print(f'auc score: {auc}')

plt.figure(figsize=(10,6))
fpr, tpr, thresholds = roc_curve(y_chur_test, y_prob)
plt.plot(fpr, tpr, label = 'roc curve')
plt.legend()
plt.show()

mse = mean_squared_error(y_card_test, y_card_pred)
mae = mean_absolute_error(y_card_test, y_card_pred)
r2 = r2_score(y_card_test, y_card_pred)
print(f'mean squared error: {mse}')
print(f'mean absolute error: {mae}')
print(f'r2 score: {r2}')

#   ---actual vs predicted---
plt.figure(figsize=(10,6))   
plt.scatter(y_card_test, y_card_test, color = 'blue', s = 5, label = 'Actual')
plt.scatter(y_card_test, y_card_pred, color = 'red', s = 5, label = 'Predicted')
plt.title('Actual vs Predicted Credit Card Limit')
plt.legend()
plt.show()

#   ---new unseen user data---
Customer_Age = int(input("\nWhat is customer's age?\n >Your Answer: "))
Gender = input("\nWhat is customer's Gender?\n >Enter M for Male\n >Enter F for Female\n >Your Answer: ")
Dependent_count = int(input("\nWhat is customer's dependent count?\n >Your Answer: "))
Education_Level = input("\nWhat is customer's education level?\n >Uneducated\n >High School\n >College\n >Graduate\n >Post-Graduate\n >Doctorate\n >Unknown\n >Your Answer: ")
Income_Category = input("\nWhat is customer's income catgeory?\n >Less than $40K\n >$40K - $60K\n >$60K - $80K\n >$80K - $120K\n >$120K +\n >Unknown\n >Your Answer: ")
Months_on_book = int(input("\nWhat is customer's months on book?\n >Your Answer: "))
Total_Relationship_Count = int(input("\nWhat is customer's total relationship count?\n >Your Answer: "))
Months_Inactive_12_mon = int(input("\nWhat is customer's months inactive in the last 12 months?\n >Your Answer: "))
Contacts_Count_12_mon = int(input("\nWhat is customer's contacts count in the last 12 months?\n >Your Answer: "))
Total_Revolving_Bal = float(input("\nWhat is customer's total revolving balance?\n >Your Answer: "))
Avg_Open_To_Buy = float(input("\nWhat is customer's average open to buy?\n >Your Answer: "))
Total_Amt_Chng_Q4_Q1 = float(input("\nWhat is customer's total amount change from Q4 to Q1?\n >Your Answer: "))
Total_Trans_Amt = float(input("\nWhat is customer's total transaction amount?\n >Your Answer: "))
Total_Trans_Ct = int(input("\nWhat is customer's total transaction count?\n >Your Answer: "))
Total_Ct_Chng_Q4_Q1 = float(input("\nWhat is customer's total count change from Q4 to Q1?\n >Your Answer: "))
Avg_Utilization_Ratio = float(input("\nWhat is customer's average utilization ratio?\n >Your Answer: "))
Marital_Status = input("\nWhat is customer's martial status?\n >Divorced\n >Married\n >Single\n >Unknown\n >Your Answer: ")
Card_Category = input("\nWhat is customer's card category?\n >Blue\n >Gold\n >Platinum\n >Silver\n >Your Answer: ")

new_chur_data = {
    "Customer_Age":[Customer_Age],
    "Gender":[Gender],
    "Dependent_count":[Dependent_count],
    "Education_Level":[Education_Level],
    "Income_Category":[Income_Category],
    "Months_on_book":[Months_on_book],
    "Total_Relationship_Count":[Total_Relationship_Count],
    "Months_Inactive_12_mon":[Months_Inactive_12_mon],
    "Contacts_Count_12_mon":[Contacts_Count_12_mon],
    "Total_Revolving_Bal":[Total_Revolving_Bal],
    "Avg_Open_To_Buy":[Avg_Open_To_Buy],
    "Total_Amt_Chng_Q4_Q1":[Total_Amt_Chng_Q4_Q1],
    "Total_Trans_Amt":[Total_Trans_Amt],
    "Total_Trans_Ct":[Total_Trans_Ct],
    "Total_Ct_Chng_Q4_Q1":[Total_Ct_Chng_Q4_Q1],
    "Avg_Utilization_Ratio":[Avg_Utilization_Ratio],
    "Marital_Status":[Marital_Status],
    "Card_Category":[Card_Category]
}
new_chur_df = pd.DataFrame(new_chur_data)

new_card_data = {
    "Customer_Age":[Customer_Age],
    "Gender":[Gender],
    "Dependent_count":[Dependent_count],
    "Education_Level":[Education_Level],
    "Income_Category":[Income_Category],
    "Months_on_book":[Months_on_book],
    "Total_Relationship_Count":[Total_Relationship_Count],
    "Months_Inactive_12_mon":[Months_Inactive_12_mon],
    "Contacts_Count_12_mon":[Contacts_Count_12_mon],
    "Total_Revolving_Bal":[Total_Revolving_Bal],
    "Total_Amt_Chng_Q4_Q1":[Total_Amt_Chng_Q4_Q1],
    "Total_Trans_Amt":[Total_Trans_Amt],
    "Total_Trans_Ct":[Total_Trans_Ct],
    "Total_Ct_Chng_Q4_Q1":[Total_Ct_Chng_Q4_Q1],
    "Avg_Utilization_Ratio":[Avg_Utilization_Ratio],
    "Marital_Status":[Marital_Status],
    "Card_Category":[Card_Category]
}
new_card_df = pd.DataFrame(new_card_data)

#   ---feature engineering---
new_card_df['Gender'] = new_card_df['Gender'].map({'M':0,'F':1})
new_chur_df['Gender'] = new_chur_df['Gender'].map({'M':0,'F':1})

new_card_df['Education_Level'] = new_card_df['Education_Level'].map({
    'Uneducated': 0,
    'High School': 1,
    'College': 2,
    'Graduate': 3,
    'Post-Graduate': 4,
    'Doctorate': 5,
    'Unknown': -1   
})
new_chur_df['Education_Level'] = new_chur_df['Education_Level'].map({
    'Uneducated': 0,
    'High School': 1,
    'College': 2,
    'Graduate': 3,
    'Post-Graduate': 4,
    'Doctorate': 5,
    'Unknown': -1   
})

new_card_df['Income_Category'] = new_card_df['Income_Category'].map({
    'Less than $40K': 0,
    '$40K - $60K': 1,
    '$60K - $80K': 2,
    '$80K - $120K': 3,
    '$120K +': 4,
    'Unknown': -1
})
new_chur_df['Income_Category'] = new_chur_df['Income_Category'].map({
    'Less than $40K': 0,
    '$40K - $60K': 1,
    '$60K - $80K': 2,
    '$80K - $120K': 3,
    '$120K +': 4,
    'Unknown': -1
})

new_chur_df = pd.get_dummies(new_chur_df, columns=['Marital_Status','Card_Category'])
new_card_df = pd.get_dummies(new_card_df, columns=['Marital_Status','Card_Category'])

new_chur_df = new_chur_df.reindex(columns=X_chur.columns, fill_value=0)
new_card_df = new_card_df.reindex(columns=X_card.columns, fill_value=0)

print(new_chur_df.isnull().sum())
print(new_chur_df.isna().sum())
print(new_card_df.isnull().sum())
print(new_card_df.isna().sum())

#   ---prediction---
prob_churn = model_chur.predict_proba(new_chur_df)[:,1]
prob_not_churn = model_chur.predict_proba(new_chur_df)[:,0]

per_churn = prob_churn[0] * 100
per_not_churn = prob_not_churn[0] * 100

pred_chur = model_chur.predict(new_chur_df)

if pred_chur[0] == 0:
    print(f"\nThe Customer is {per_not_churn}% likely to remain an existing customer")
else:
    print(f"The Customer is {per_churn:.2f}% at risk of churning")



pred_card = model_card.predict(new_card_df)
print(f"The Customer's predicted credit card limit is ${pred_card[0]}")