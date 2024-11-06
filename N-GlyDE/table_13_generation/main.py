from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,f1_score,matthews_corrcoef,average_precision_score
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import pickle
import csv


def preprocess_the_dataset(feature_X):

    pt = PowerTransformer()
    pt.fit(feature_X)
    feature_X = pt.transform(feature_X)

    return feature_X


def make_string(s):
    str = ''
    for i in s:
        str += i + ", "
    return str[:-2]


def load_the_pickle_files_base_layer(converted_all_features_with_output):
    test_X = preprocess_the_dataset(converted_all_features_with_output)
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
    test_X = preprocess_the_dataset(converted_all_features_with_output)

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


feature_paths = {
    'ProtT5-XL-U50-Per-Residue': '/media/nafiislam/New Volume/For side thesis/all_features/ProtT5-XL-U50-Per-Residue/fasta_test_NGlyDE_original.csv',
    'ESM-2-Window': '/media/nafiislam/New Volume/For side thesis/all_features/ESM-2-Window/fasta_test_NGlyDE_original.csv',
    'ProteinBert': '/media/nafiislam/New Volume/For side thesis/all_features/ProteinBert/fasta_test_NGlyDE_original.csv',
}

with open('./results.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Predictor', 'Sensitivity', 'Specificity', 'Balanced_acc', 'Accuracy', 'Precision', 'F1-score', 'MCC', 'AUC', 'auPR'])

    file_path_Benchmark_embeddings = feature_paths['ProteinBert']
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
    feature_y_Benchmark = D_feature.iloc[:, 0].values

    feature_X_Benchmark = np.zeros((feature_y_Benchmark.shape[0], 1), dtype=float)

    file_path_Benchmark_embeddings = feature_paths['ProteinBert']
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
    feature_X_Benchmark = np.concatenate((feature_X_Benchmark, D_feature.iloc[:, 2:].values), axis=1)

    file_path_Benchmark_embeddings = feature_paths['ESM-2-Window']
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
    feature_X_Benchmark = np.concatenate((feature_X_Benchmark, D_feature.iloc[:, 2:].values), axis=1)

    file_path_Benchmark_embeddings = feature_paths['ProtT5-XL-U50-Per-Residue']
    D_feature = pd.read_csv(file_path_Benchmark_embeddings, header=None, low_memory=False)
    feature_X_Benchmark = np.concatenate((feature_X_Benchmark, D_feature.iloc[:, 2:].values), axis=1)

    feature_X_Benchmark = np.delete(feature_X_Benchmark, 0, axis=1)

    X = feature_X_Benchmark.copy()
    X = preprocess_the_dataset(X)
    y = feature_y_Benchmark.copy()

    print('X : ', X.shape)
    print('y : ', y.shape)

    BLP = load_the_pickle_files_base_layer(X)
    X = np.concatenate((X, BLP), axis=1)

    print('X : ', X.shape)
    print('y : ', y.shape)

    y_pred, y_proba = load_the_pickle_files_meta_layer(X)

    sensitivity, specificity, bal_acc, acc, prec, f1_score_1, mcc, auc, auPR = find_metrics(y_pred, y_proba, y)

    print('Sensitivity : {0:.3f}'.format(sensitivity))
    print('Specificity : {0:.3f}'.format(specificity))
    print('Balanced_acc : {0:.3f}'.format(bal_acc))
    print('Accuracy : {0:.3f}'.format(acc))
    print('Precision : {0:.3f}'.format(prec))
    print('F1-score: {0:.3f}'.format(f1_score_1))
    print('MCC: {0:.3f}'.format(mcc))
    print('AUC: {0:.3f}'.format(auc))
    print('auPR: {0:.3f}'.format(auPR))

    writer.writerow(['StackGlyEmbed', '{0:.3f}'.format(sensitivity), '{0:.3f}'.format(specificity), '{0:.3f}'.format(bal_acc), '{0:.3f}'.format(acc), '{0:.3f}'.format(prec), '{0:.3f}'.format(f1_score_1), '{0:.3f}'.format(mcc), '{0:.3f}'.format(auc), '{0:.3f}'.format(auPR)])