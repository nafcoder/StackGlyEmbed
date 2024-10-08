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

df_test = pd.read_csv("df_indepenent_test_again_done_that_has_unique_protein_and_unique_sequence.csv")

print(df_test["label"].value_counts())

y_independent = np.array(df_test["label"])

df_test = df_test.iloc[:,5:]
X_independent = np.array(df_test)

print(X_independent.shape,y_independent.shape)

model = tf.keras.models.load_model("Final_GlycoBiology_ANN_Glycobiology_ER_RSA(GA_Extracell_cellmem)187.h5")

print(model.summary())

Y_pred = model.predict(X_independent)
Y_pred = (Y_pred > 0.5)
y_pred = [np.argmax(y, axis=None, out=None) for y in Y_pred]
y_pred = np.array(y_pred)

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