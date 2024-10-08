import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import *
from sklearn.metrics import roc_curve, roc_auc_score, classification_report,auc
# matplotlib inline
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from keras.layers import Dense, Bidirectional,Dense, LSTM, Activation, Dropout, Flatten
from keras.layers import LeakyReLU
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.regularizers import l2
from tensorflow.keras import regularizers
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential



df_test = pd.read_csv("All_feature_Independent_dataframe.csv")

X_test_NetSurfP2 = df_test.iloc[:,2100:2428]

y_test_NetSurfP2 = df_test["Target"]


X_test_NetSurfP2.fillna(0,inplace=True)

scaler = MinMaxScaler()


X_test_NetSurfP2 = scaler.fit_transform(X_test_NetSurfP2)


X_test_NetSurfP2 = np.array(X_test_NetSurfP2)
y_test_NetSurfP2 = np.array(y_test_NetSurfP2)

X_test_PSSM_test = df_test.iloc[:,1280:2100]

y_test_PSSM = df_test["Target"]

X_test_PSSM_test.fillna(0,inplace=True)

scaler = MinMaxScaler()


X_test_PSSM_test = scaler.fit_transform(X_test_PSSM_test)


X_test_PSSM_test = np.array(X_test_PSSM_test)
y_test_PSSM = np.array(y_test_PSSM)

X_test_Gapped_Dipeptide = df_test.iloc[:,2428:2468]

y_test_Gapped_Dipeptide = df_test["Target"]


scaler = MinMaxScaler()


X_test_Gapped_Dipeptide = scaler.fit_transform(X_test_Gapped_Dipeptide)


X_test_Gapped_Dipeptide = np.array(X_test_Gapped_Dipeptide)
y_test_Gapped_Dipeptide = np.array(y_test_Gapped_Dipeptide)

X_test_PSSM_NetSurfP2_Gapped_Dipeptide = np.hstack((X_test_PSSM_test,X_test_NetSurfP2,X_test_Gapped_Dipeptide))

y_test_PSSM_NetSurfP2_Gapped_Dipeptide = y_test_Gapped_Dipeptide

X_test = X_test_PSSM_NetSurfP2_Gapped_Dipeptide

y_test = y_test_PSSM_NetSurfP2_Gapped_Dipeptide


model = keras.models.load_model("Glycosylation_model__26__.h5")
Y_pred = model.predict(X_test)
DNN_prediction_list = []
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)
result_obtained = []
y_pred = [np.argmax(y, axis=None, out=None) for y in Y_pred]

y_pred = np.array(y_pred)
print("Matthews Correlation : ",matthews_corrcoef(y_test, y_pred))
print("\n")
print("Confusion Matrix : \n\n",confusion_matrix(y_test, y_pred))
print("\n")
print("Accuracy on test set:   ",accuracy_score(y_test, y_pred))
print("\n")


cm = confusion_matrix(y_test, y_pred)

TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]

mcc = matthews_corrcoef(y_test, y_pred)


Sensitivity = TP/(TP+FN)

Specificity = TN/(TN+FP)

Precision = TP/(TP+FP)
print("Sensitivity:   ",Sensitivity,"\t","Specificity:   ",Specificity)
print("\n")
print("Precision:   ",Precision)
print("\n")

print(classification_report(y_test, y_pred))
