import pandas as pd 
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,f1_score,matthews_corrcoef,average_precision_score
from sklearn.metrics import balanced_accuracy_score


def find_metrics(y_predict, y_proba, y_test):

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()  # y_true, y_pred

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    bal_acc = balanced_accuracy_score(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)

    if tp == 0 and fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)

    if prec == 0 and sensitivity == 0:
        f1_score_1 = 0
    else:
        f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    auPR = average_precision_score(y_test, y_proba[:, 1])  # auPR

    return sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR


test = pd.read_excel("447_feature_test.xlsx")

# NetSurfP2.0
X_test_Net = test.iloc[:,1783:1983]
y_test_Net = test["Target"]


print(X_test_Net.isnull().sum().sum())
X_test_Net.fillna(0,inplace=True)
print(X_test_Net.isnull().sum().sum())

scaler = MinMaxScaler()


X_test_Net = scaler.fit_transform(X_test_Net)


X_test_Net = np.array(X_test_Net)
y_test_Net = np.array(y_test_Net)

# Gapped Dipeptide Features
X_test_GD = test.iloc[:,1983:2007]
y_test_GD = test["Target"]


scaler = MinMaxScaler()


X_test_GD = scaler.fit_transform(X_test_GD)


X_test_GD = np.array(X_test_GD)
y_test_GD = np.array(y_test_GD)

# PSSM Features
X_test_PSSM = test.iloc[:,1283:1783]
y_test_PSSM = test["Target"]


print(X_test_PSSM.isnull().sum().sum())
X_test_PSSM.fillna(0,inplace=True)
print(X_test_PSSM.isnull().sum().sum())

scaler = MinMaxScaler()


X_test_PSSM = scaler.fit_transform(X_test_PSSM)


X_test_PSSM = np.array(X_test_PSSM)
y_test_PSSM = np.array(y_test_PSSM)

# Combine Three Features (NetSurfP-2.0, Gapped Dipeptide, PSSM) and Check the size of the matrix, shuffle it
X_test_Net_PSSM_GD = np.hstack((X_test_Net,X_test_PSSM,X_test_GD))

y_test_Net_PSSM_GD = y_test_Net


X_test, y_test = shuffle(X_test_Net_PSSM_GD, y_test_Net_PSSM_GD, random_state=7)
X_test = np.array(X_test)
y_test = np.array(y_test)


print(X_test.shape,y_test.shape)

# Load the model and see the performance matrices
model = keras.models.load_model("3080_NetSurfP2.0_PSSM_GD"+str(8)+".h5")
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)
y_pred = [np.argmax(y, axis=None, out=None) for y in Y_pred]
y_pred = np.array(y_pred)

sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(y_pred, Y_pred, y_test)

print('Sensitivity : {0:.3f}'.format(sensitivity))
print('Specificity : {0:.3f}'.format(specificity))
print('Balanced_acc : {0:.3f}'.format(bal_acc))
print('Accuracy : {0:.3f}'.format(acc))
print('Precision : {0:.3f}'.format(prec))
print('F1-score: {0:.3f}'.format(f1_score_1))
print('MCC: {0:.3f}'.format(mcc))
print('AUC: {0:.3f}'.format(auc))
print('auPR: {0:.3f}'.format(auPR))