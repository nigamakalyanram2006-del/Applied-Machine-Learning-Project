#   ---Predicting whether or not the borrower paid back their loan in full, using Logistic Regression ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

#   ---loading dataset---
df = pd.read_csv('loan_data (1).csv')
print(df)

#   ---checking for missing values---
print(df.isnull().sum())

#   ---feature engineering---
pd.set_option('display.max_columns', None)
print(df.head())

print(df['purpose'].unique())
df['purpose'] = df['purpose'].map({'debt_consolidation':0, 'credit_card':1, 'all_other':2, 'home_improvement':3, 'small_business':4, 'major_purchase':5, 'educational':6})
print(df)
print(df.isnull().sum())
print(df.isna().sum())

#   ---separating features and target---
X = df.drop('not.fully.paid', axis = 1)
y = df['not.fully.paid']

#   ---creating testing and training sets---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

#   ---standardization---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#   ---model initiation and training---
model = LogisticRegression(C=0.01, max_iter=1000)
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
cr = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cr)
print(cm)
cmdisp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['No','Yes'])
cmdisp.plot()
plt.show()

#   ---auc score and roc curve---
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)
print(f'auc score: {auc_score}')

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label = 'roc curve')
plt.legend()
plt.show()

#   ---new, unseen user data--- credit.policy,	purpose,	int.rate,	installment,	log.annual.inc,	dti,	fico,	days.with.cr.line,	revol.bal,	revol.util,	inq.last.6mths,	delinq.2yrs,	pub.rec
credit_policy = int(input("\nMeets credit policy?\n >Enter 0 for No\n >Enter 1 for Yes\n >Your Answer: "))
purpose = int(input('\nEnter Loan Purpose\n >Enter 0 for Debt Consolidation\n >Enter 1 for Credit Card\n >Enter 2 for All Other\n >Enter 3 for Home Improvement\n >Enter 4 for Small Business\n >Enter 5 for Major Purchase\n >Enter 6 for Educational\n >Your Answer: '))
int_rate = float(input("\nEnter interest rate as decimal (e.g., 0.12 for 12%)\n >Your Answer: "))
installment = float(input("\nEnter monthly installment amount in dollars\n >Your Answer: "))
log_annual_inc = float(input("\nEnter log of annual income\n >Your Answer: "))
dti = float(input(
"\nWhat is your debt-to-income ratio? (percentage of income used to pay debts)\n >Your Answer: "
))
fico = int(input("\nEnter your FICO credit score (300-850)\n >Your Answer: "))
days_with_cr_line = float(input(
"\nHow many days have you had credit history?\n >Your Answer: "
))
revol_bal = float(input(
"\nWhat is your total revolving credit balance (e.g., credit cards) in dollars?\n >Your Answer: "
))
revol_util = float(input(
"\nWhat percentage of your available revolving credit are you using?\n >Your Answer: "
))
inq_last_6mths = int(input(
"\nHow many credit inquiries have you had in the last 6 months?\n >Your Answer: "
))
delinq_2yrs = int(input(
"\nHow many times have you been delinquent on a payment in the past 2 years?\n >Your Answer: "
))
pub_rec = int(input(
"\nHow many public records (bankruptcies, liens, etc.) do you have?\n >Your Answer: "
))

user_data = {
    "credit.policy": [credit_policy],
    "purpose": [purpose],
    "int.rate": [int_rate],
    "installment": [installment],
    "log.annual.inc": [log_annual_inc],
    "dti": [dti],
    "fico": [fico],
    "days.with.cr.line": [days_with_cr_line],
    "revol.bal": [revol_bal],
    "revol.util": [revol_util],
    "inq.last.6mths": [inq_last_6mths],
    "delinq.2yrs": [delinq_2yrs],
    "pub.rec": [pub_rec]
}
user_df = pd.DataFrame(user_data)
print(user_df)

#   ---standardization---
user_df = scaler.transform(user_df)

#   ---prediction and probability---
pred = model.predict(user_df)

prob_yes = model.predict_proba(user_df)[:,1]
prob_no = model.predict_proba(user_df)[:,0]

per_yes = prob_yes * 100
per_no = prob_no * 100

if pred[0] == 1:
  print(f'There is a {per_yes[0]}% chance the Borrower will NOT pay their Loan back')
else:
  print(f'There is a {per_no[0]}% chance the Borrower WILL pay their Loan back')