#   ---Predicting Breast Cancer using SVC ML Model---

#   ---importing libraries---
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

#   ---loading dataset---
df = pd.read_csv('breastcancer_data.csv')

#   ---dropping NaN values---
print(df.isnull().sum())
print(df.isna().sum())
df = df.drop('Unnamed: 32', axis = 1)
print(df)

#   ---feature engineering---
df = df.drop('id', axis = 1)
pd.set_option('display.max_columns', None)
print(df.head())

print(df['diagnosis'].unique())
df['diagnosis'] = df['diagnosis'].map({'M':0, 'B':1})
print(df.isnull().sum())

#   ---separating features and target---
X = df.drop('diagnosis', axis = 1)
y = df['diagnosis']

#  ---creating testing and training sets---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

#   ---standardization---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#   ---model initiation and training---
model  = SVC(kernel = "linear", C = 1.0, probability=True)
model.fit(X_train, y_train)

#   ---prediction---
y_pred = model.predict(X_test)

#   ---model evaluation---
cr = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cr)
print(cm)

cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Malignant', 'Benign'])
cmd.plot()
plt.show()

y_prob = model.predict_proba(X_test)[:,1]
auc_score = roc_auc_score(y_test, y_prob)
print(auc_score)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label = "roc curve")
plt.show()

#   ---new, unseen user inputs---
user_data = [
    # 1. Clear Malignant
    [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
    # 2. Clear Benign
    [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259],
    # 3. Malignant
    [20.57, 17.77, 132.9, 1326.0, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, 0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532, 24.99, 23.41, 158.8, 1956.0, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902],
    # 4. Benign
    [13.08, 15.71, 85.63, 520.0, 0.1075, 0.127, 0.04568, 0.0311, 0.1967, 0.06811, 0.1852, 0.7477, 1.383, 14.67, 0.004097, 0.01898, 0.01698, 0.00649, 0.01678, 0.002425, 14.5, 20.49, 96.09, 630.5, 0.1312, 0.2776, 0.189, 0.07283, 0.3184, 0.08183],
    # 5. Borderline Malignant
    [15.1, 23.0, 100.0, 700.0, 0.1, 0.15, 0.12, 0.08, 0.2, 0.06, 0.5, 1.2, 3.5, 45.0, 0.008, 0.03, 0.04, 0.02, 0.025, 0.005, 19.0, 30.0, 130.0, 1100.0, 0.15, 0.35, 0.45, 0.18, 0.35, 0.1],
    # 6. Benign
    [12.31, 16.52, 79.19, 470.9, 0.09172, 0.06829, 0.03372, 0.02272, 0.172, 0.05914, 0.2505, 1.025, 1.74, 19.68, 0.004854, 0.01819, 0.01823, 0.008824, 0.01906, 0.002498, 14.11, 23.21, 89.6, 612.9, 0.1188, 0.1747, 0.1922, 0.09444, 0.2808, 0.07413],
    # 7. Malignant
    [19.17, 24.8, 132.4, 1123.0, 0.0974, 0.2458, 0.2065, 0.1118, 0.2397, 0.078, 0.9555, 3.568, 11.07, 116.2, 0.003139, 0.08297, 0.0889, 0.0409, 0.04484, 0.01284, 20.96, 29.94, 151.7, 1332.0, 0.1037, 0.3903, 0.3639, 0.1767, 0.3176, 0.1023],
    # 8. Benign
    [11.42, 20.38, 77.58, 386.1, 0.1425, 0.2839, 0.2414, 0.1052, 0.2597, 0.09744, 0.4956, 1.156, 3.445, 27.23, 0.00911, 0.07458, 0.05661, 0.01867, 0.05963, 0.009208, 14.91, 26.5, 98.87, 567.7, 0.2098, 0.8663, 0.6869, 0.2575, 0.6638, 0.173],
    # 9. Malignant
    [14.99, 21.19, 97.9, 712.8, 0.115, 0.1642, 0.1574, 0.08039, 0.1857, 0.06492, 0.5433, 1.595, 3.443, 53.15, 0.00578, 0.02647, 0.02908, 0.01046, 0.01768, 0.003136, 18.07, 29.06, 119.4, 1002.0, 0.1444, 0.4245, 0.4504, 0.243, 0.3135, 0.1232],
    # 10. Benign
    [13.03, 18.42, 82.61, 523.8, 0.08983, 0.03766, 0.02562, 0.02923, 0.1467, 0.05863, 0.1839, 2.342, 1.17, 14.16, 0.004352, 0.004899, 0.01343, 0.01164, 0.02671, 0.001777, 13.3, 22.81, 84.46, 545.9, 0.09701, 0.04619, 0.04833, 0.05013, 0.1987, 0.06169],
    # 11. Malignant
    [15.78, 17.89, 103.6, 781.0, 0.0971, 0.1292, 0.09954, 0.06606, 0.1842, 0.06082, 0.5058, 0.9849, 3.564, 54.16, 0.005771, 0.04061, 0.02791, 0.01282, 0.02008, 0.004144, 20.42, 27.28, 136.5, 1299.0, 0.1396, 0.5609, 0.3965, 0.181, 0.3792, 0.1048],
    # 12. Benign
    [12.45, 15.7, 82.57, 477.1, 0.1278, 0.17, 0.1578, 0.08089, 0.2087, 0.07613, 0.3345, 0.8902, 2.217, 27.19, 0.00751, 0.03345, 0.03672, 0.01137, 0.02165, 0.005082, 15.47, 23.75, 103.4, 741.6, 0.1791, 0.5249, 0.5355, 0.1741, 0.3985, 0.1244],
    # 13. Malignant
    [19.81, 22.15, 130.0, 1260.0, 0.09831, 0.1027, 0.1479, 0.09498, 0.193, 0.05539, 0.7862, 1.323, 5.839, 94.44, 0.004466, 0.02226, 0.02545, 0.01031, 0.01383, 0.002708, 27.32, 30.88, 186.8, 2398.0, 0.1512, 0.315, 0.5372, 0.2388, 0.2768, 0.07615],
    # 14. Benign
    [9.504, 12.44, 60.34, 273.9, 0.1024, 0.06492, 0.02956, 0.02076, 0.1815, 0.06905, 0.2773, 0.9768, 1.909, 15.7, 0.009606, 0.01432, 0.01985, 0.01421, 0.02027, 0.002968, 10.23, 15.66, 65.13, 314.9, 0.1324, 0.1148, 0.08867, 0.06227, 0.245, 0.07773],
    # 15. Borderline Benign
    [14.5, 18.0, 95.0, 650.0, 0.09, 0.1, 0.08, 0.04, 0.18, 0.06, 0.3, 1.0, 2.5, 30.0, 0.006, 0.02, 0.02, 0.01, 0.02, 0.003, 16.0, 22.0, 105.0, 800.0, 0.12, 0.2, 0.25, 0.1, 0.28, 0.07]
]

user_data = scaler.transform(user_data)

for i in range(len(user_data)):
    user_pred = model.predict(user_data[i:i+1])
    user_prob = model.predict_proba(user_data[i:i+1])
    
    user_pred_m_per = user_prob[0][0] * 100
    user_pred_b_per = user_prob[0][1] * 100
    
    if user_pred[0] == 0:
        print(f"User {i+1} Diagnosis: Malignant ({user_pred_m_per}% confident)")
    else:
        print(f"User {i+1} Diagnosis: Benign ({user_pred_b_per}% confident)")