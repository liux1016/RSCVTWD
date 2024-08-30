import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def group_indices(y):
    result = {}
    for index, target in y.items():
        if target not in result:
            result[target] = set()
        result[target].add(index)
    return result



def group_indices_by_features(X):
    result = {}
    for index, row in X.iterrows():
        key = '_'.join(str(value) for value in row.values)
        if key not in result:
            result[key] = set()
        result[key].add(index)
    return result



def calProbability(D_i, R_j):
    intersection_length = len(D_i & R_j)
    probability = intersection_length / len(R_j)
    return probability



def divThreeArea(prob, a, b, R_j_key, POS, BND, NEG):
    if prob >= a:
        POS.append(R_j_key)
    elif prob <= b:
        NEG.append(R_j_key)
    else:
        BND.append(R_j_key)
    return POS, BND, NEG



def predict(X_test, POS, BND, NEG, D_i_key):
    y_pred = []
    for index, row in X_test.iterrows():
        key = '_'.join(str(value) for value in row.values)
        if key in POS:
            y_pred.append(D_i_key)
        elif key in NEG:
            y_pred.append(-1)
        else:
            y_pred.append(-2)
    return y_pred


def calculate_f1_score(y_test, y_pred, D_i_key):
    TP = sum(1 for true_val, pred_val in zip(y_test, y_pred) if true_val == D_i_key and pred_val == D_i_key)
    FP = sum(1 for true_val, pred_val in zip(y_test, y_pred) if true_val != D_i_key and pred_val == D_i_key)
    FN = sum(1 for true_val, pred_val in zip(y_test, y_pred) if true_val == D_i_key and pred_val == -1)
    if TP == 0:
        f1_score = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

start = time.time()
#TODO
data = pd.read_csv('dis_iris.csv')
K = 10
a = 0.7
b = 0.3
#
l=1
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
skf = StratifiedKFold(n_splits=K)
results_X = {}
results_y = {}
results_X_input = {}
results_y_input = {}
F1 = {}
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    results_y = group_indices(y_train)
    results_X = group_indices_by_features(X_train)
    for D_i_key in results_y.keys():
        POS = []
        BND = []
        NEG = []
        for R_j_key in results_X.keys():
            probability= calProbability(results_y.get(D_i_key), results_X.get(R_j_key))
            divThreeArea(probability, a, b, R_j_key, POS, BND, NEG)
        y_pred = predict(X_test, POS, BND, NEG, D_i_key)
        F1[f"{l}--{D_i_key}"] = calculate_f1_score(y_test, y_pred,D_i_key)
    l += 1
F1_sum = sum(F1.values())
F1_avg = F1_sum / len(F1)
end = time.time()
print("F1:", F1)
print("F1_avg=",F1_avg)
print("time:", end - start)
