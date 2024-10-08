import pandas as pd
import numpy as np
import os
import tensorflow as tf
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


model = tf.keras.models.load_model("Undersampling_Glycobiology_NGLYDE_Final6947757.h5")

print(model.summary())



Header_name = ["Label","PID","POsition","Sequence","Middle_Amino_Acid_ASN(N)"]

col_of_feature = [i for i in range(1,1025)]

Header_name = Header_name + col_of_feature

df_test = pd.read_csv("Independent_Test_Set_Prot_T5_feature_Aug_12.txt", header=None)

df_test.columns = Header_name

df_test_123 = df_test

df_test = df_test.iloc[:,5:]
X_independent = np.array(df_test)

y_test_indi_positive = [1]*166
y_test_indi_negative = [0]*(444-166)
y_independent = y_test_indi_positive+y_test_indi_negative
y_independent = np.array(y_independent)
print(len(y_independent))

print(X_independent.shape,y_independent.shape)

print(df_test_123["Label"].value_counts())

Y_pred = model.predict(X_independent)
Y_pred = (Y_pred > 0.5)
y_pred = [np.argmax(y, axis=None, out=None) for y in Y_pred]
y_pred = np.array(y_pred)
print(y_pred.shape)
sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(y_pred, Y_pred, y_independent)

print('Sensitivity : {0:.3f}'.format(sensitivity))
print('Specificity : {0:.3f}'.format(specificity))
print('Balanced_acc : {0:.3f}'.format(bal_acc))
print('Accuracy : {0:.3f}'.format(acc))
print('Precision : {0:.3f}'.format(prec))
print('F1-score: {0:.3f}'.format(f1_score_1))
print('MCC: {0:.3f}'.format(mcc))
print('AUC: {0:.3f}'.format(auc))
print('auPR: {0:.3f}'.format(auPR))