#   ---Credit Card Segmentation using K-Means Clustering ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#   ---loading dataset---
df = pd.read_csv('CC GENERAL.csv')

#   ---checking for missing values---
print(df.isnull().sum())

#   ---filling in missing values---
df.dropna(subset=['CREDIT_LIMIT'], inplace=True)
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())

#   ---feature engineering---
df = df.drop('CUST_ID', axis = 1)

#   ---features---
X = df

#   ---PCA---
scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components = 3, random_state = 101)
X = pca.fit_transform(X)

#   ---initiating model---
wcss = []
for i in range(1, 11):
  model = KMeans(n_clusters = i, random_state = 101)
  model.fit(X)
  wcss.append(model.inertia_)
print(wcss)

#   ---plotting wcss and using elbow method---
plt.plot(range(1, 11), wcss)
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()

model = KMeans(n_clusters = 4, random_state = 101)
model.fit(X)

#   ---adding clusters to dataframe---
df['clusters'] = model.labels_
print(df)

#   ---visualizing the clusters and centeroids---
plt.figure(figsize = (10, 6))
plt.scatter(X[:,0], X[:,1], c = df['clusters'])
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker = '*', s = 200, color = 'black')
plt.show()

#   ---interpreting---
print(df.groupby('clusters').mean())

cluster_meaning = {
  0: "Moderate/Everyday Users\nBalanced/Responsible Average users, spend moderately, often pay in full",
  1: "High-Value/Heavy Spenders\nBig spenders, high credit limits, pay a lot back",
  2: "Cash Advance/Debt-Heavy Users\nNot spending much on purchases, likely using cash advances, carrying large balances (debt)",
  3: "Inactive/Low Useage Customers\nLow and Infrequent Activity"
}

#   ---since the model is good, we will use it on new, unseen user data---
balance = float(input("\nWhat is your account balance? "))
balance_frequency = float(input("\nHow often is your balance updated? (1.0 = monthly, 0.0 = never): "))
purchases = float(input("\nWhat is the total value of all purchases made? (Add up the dollar amount of every purchase you’ve made using the card) "))
oneoff_purchases = float(input("\nWhat is the total amount spent on one-time, single transactions? "))
installments_purchases = float(input("\nWhat is the total amount spent on purchases made in installments? "))
cash_advance = float(input("\nWhat is the total amount of cash withdrawn from the card? "))
purchases_frequency = float(input("\nHow often do you make purchases? (1.0 = every month, 0.0 = never): "))
oneoff_purchases_frequency = float(input("\nHow frequently do you make one-time purchases? (0.0 to 1.0): "))
purchases_installments_frequency = float(input("\nHow frequently do you make installment-based purchases? (0.0 to 1.0): "))
cash_advance_frequency = float(input("\nHow often do you take cash advances? (0.0 to 1.0): "))
cash_advance_trx = int(input("\nHow many times have you used your card to withdraw cash? "))
purchases_trx = int(input("\nHow many separate purchase transactions have you made? "))
credit_limit = float(input("\nWhat is the maximum credit limit allowed on the card? "))
payments = float(input("\nWhat is the total amount of payments made toward the balance? "))
minimum_payments = float(input("\nWhat is the total amount of 'minimum payments' made? "))
prc_full_payment = float(input("\nWhat percentage of the full statement do you pay off? (0.0 to 1.0): "))
tenure = int(input("\nHow many months has the card been active? (6 to 12): "))

new_data = {
    'BALANCE': [balance],
    'BALANCE_FREQUENCY': [balance_frequency],
    'PURCHASES': [purchases],
    'ONEOFF_PURCHASES': [oneoff_purchases],
    'INSTALLMENTS_PURCHASES': [installments_purchases],
    'CASH_ADVANCE': [cash_advance],
    'PURCHASES_FREQUENCY': [purchases_frequency],
    'ONEOFF_PURCHASES_FREQUENCY': [oneoff_purchases_frequency],
    'PURCHASES_INSTALLMENTS_FREQUENCY': [purchases_installments_frequency],
    'CASH_ADVANCE_FREQUENCY': [cash_advance_frequency],
    'CASH_ADVANCE_TRX': [cash_advance_trx],
    'PURCHASES_TRX': [purchases_trx],
    'CREDIT_LIMIT': [credit_limit],
    'PAYMENTS': [payments],
    'MINIMUM_PAYMENTS': [minimum_payments],
    'PRC_FULL_PAYMENT': [prc_full_payment],
    'TENURE': [tenure]
}

new_df = pd.DataFrame(new_data)
print(new_df)

new_df = scaler.transform(new_df)
new_df = pca.transform(new_df)

#   ---prediction---
pred = model.predict(new_df)[0]

print(f'This person falls under the category of: {cluster_meaning[pred]}')