from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,f1_score,matthews_corrcoef,average_precision_score
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import pickle


def preprocess_the_dataset(feature_X, i):
    file = f'power_transformer_{i}.sav'
    with open(file, 'rb') as f:
        pt = pickle.load(f)
    feature_X = pt.transform(feature_X)

    return feature_X


def make_string(s):
    str = ''
    for i in s:
        str += i + ", "
    return str[:-2]


def load_the_pickle_files_base_layer(converted_all_features_with_output):
    test_X = preprocess_the_dataset(converted_all_features_with_output, 2)
    test_base_output_total = np.zeros((len(test_X), 1), dtype=float)

    pickle_folder_path = str('./base_layer_pickle_files/')

    base_classifiers = ['SVM', 'XGB', 'KNN']

    for i in range(0, 10):
        for base_classifier in base_classifiers:
            pickle_file_path = str(pickle_folder_path + base_classifier + '_base_layer_' + str(i) + '.sav')
            # print(pickle_file_path)

            outfile = open(pickle_file_path, 'rb')
            model = pickle.load(outfile)
            outfile.close()

            y_proba = model.predict_proba(test_X)[:, 1].reshape(-1, 1)
            test_base_output_total = np.concatenate((test_base_output_total, y_proba), axis=1)

    test_base_output_total = np.delete(test_base_output_total, 0, axis=1)

    return test_base_output_total


def load_the_pickle_files_meta_layer(converted_all_features_with_output):
    test_X = preprocess_the_dataset(converted_all_features_with_output, 3)

    pickle_file_path = str('./base_layer_pickle_files/SVM_meta_layer.sav')

    outfile = open(pickle_file_path, 'rb')
    clf = pickle.load(outfile)
    outfile.close()

    y_pred = clf.predict(test_X)
    y_prob = clf.predict_proba(test_X)

    return y_pred, y_prob


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


with open('predicted_values.txt', 'w') as f:

    D_feature = pd.read_csv('features.csv', header=None, low_memory=False)
    feature_X_Benchmark = D_feature.values
    print('feature_X_Benchmark : ', feature_X_Benchmark.shape)

    X = feature_X_Benchmark
    X = preprocess_the_dataset(X, 1)

    print('X : ', X.shape)

    BLP = load_the_pickle_files_base_layer(X)
    X = np.concatenate((X, BLP), axis=1)

    print('X : ', X.shape)

    y_pred, y_proba = load_the_pickle_files_meta_layer(X)

    st = ''
    for i in y_pred:
        st += str(i) + ','
    f.write(st[:-1] + '\n')

