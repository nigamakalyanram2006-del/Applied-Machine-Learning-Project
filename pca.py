#   ---predicting the biological origin (type) of cancer based on the gene expression levels of the patient using PCA and Random Forest ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#   ---loading dataset + separating features and targets---
X = pd.read_csv('data.csv') #features
y = pd.read_csv('labels.csv') #target

#   ---checking for missing values---
X = X.drop('Unnamed: 0', axis = 1)
print(X.isnull().sum())

y = y.drop('Unnamed: 0', axis = 1)
print(y.isnull().sum())
print(y)

#   ---feature engineering---
le = LabelEncoder()
y = le.fit_transform(y['Class'])
print(dict(zip(le.classes_, le.transform(le.classes_))))

#BRCA 0
#COAD 1
#KIRC 2
#LUAD 3
#PRAD 4

#   ---creating training and testing sets---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

#   ---PCA---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=8)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#   ---model initiation and training---
model = RandomForestClassifier(n_estimators=24, random_state=101)
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

#   ---new, unseen data---
new_patient_raw = X.iloc[0:1, :] #simulating a brand new patient
new_patient_processed = pca.transform(scaler.transform(new_patient_raw)) #model now treats the first row from the original data like it has never seen it before

pred = model.predict(new_patient_processed)

if pred[0] == 0:
    print("The model has analyzed 20,531 genes and identifies this as BRCA")
elif pred[0] == 1:
    print("The model has analyzed 20,531 genes and identifies this as COAD")
elif pred[0] == 2:
    print("The model has analyzed 20,531 genes and identifies this as KIRC")
elif pred[0] == 3:
    print("The model has analyzed 20,531 genes and identifies this as LUAD")
elif pred[0] == 4:
    print("The model has analyzed 20,531 genes and identifies this as PRAD")
else:
    print("Something went wrong, the model was not able to identify the cancer type, try again later")