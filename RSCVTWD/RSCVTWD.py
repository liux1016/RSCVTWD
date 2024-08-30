import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def group_indices(y_train):
    result = {}
    for index, target in y_train.items():
        if target not in result:
            result[target] = set()
        result[target].add(index)
    return result


def group_indices_by_features(X_train):
    result = {}
    for index, row in X_train.iterrows():
        key = '_'.join(str(value) for value in row.values)
        if key not in result:
            result[key] = set()
        result[key].add(index)
    return result


def calProbability(D_i, R_j):
    intersection_length = len(D_i & R_j)
    probability = intersection_length / len(R_j)
    return probability, intersection_length


def divThreeArea(prob, a, b, R_j_key, POS, BND, NEG):
    if prob >= a:
        POS.append(R_j_key)
    elif prob <= b:
        NEG.append(R_j_key)
    else:
        BND.append(R_j_key)
    return POS, BND, NEG


def calTSet(results_y_1, results_y_output, results_y_input):
    results_y_l = {}
    for key in results_y_1.keys():
        D_i_1 = results_y_1.get(key, set())
        D_i_output = results_y_output.get(key, set())
        D_i_input = results_y_input.get(key, set())
        D_i_l = (D_i_1 - D_i_output) | D_i_input
        results_y_l[key] = D_i_l
    return results_y_l


def calCSet(results_X_1, results_X_output, results_X_input):
    results_X_l = {}
    R = set(results_X_1) | set(results_X_input)
    for key in R:
        R_j_1 = results_X_1.get(key, set())
        R_j_output = results_X_output.get(key, set())
        R_j_input = results_X_input.get(key, set())
        R_j_l = (R_j_1 - R_j_output) | R_j_input
        if len(R_j_l) != 0:
            results_X_l[key] = R_j_l
    return results_X_l


def calNewProbability(D_i_1, D_i_output, D_i_input, R_j_1, R_j_output, R_j_input, Len):
    if len(R_j_input) == 0 and len(R_j_output) == 0:
        newPr_tag = 0
        return newPr_tag
    if len(R_j_input) == 0 and len(R_j_output) != 0:
        i = Len * len(R_j_output) - len(R_j_1) * len((D_i_output & R_j_1) | (D_i_1 & R_j_output))
        if i > 0:
            newPr_tag = 1
        elif i < 0:
            newPr_tag = -1
        else:
            newPr_tag = 0
        return newPr_tag
    if len(R_j_input) != 0 and len(R_j_output) == 0:
        i = Len * len(R_j_input) - len(R_j_1) * len(D_i_input & R_j_input)
        if i > 0:
            newPr_tag = -1
        elif i < 0:
            newPr_tag = 1
        else:
            newPr_tag = 0
        return newPr_tag
    if len(R_j_input) != 0 and len(R_j_output) != 0:
        i = Len * (len(R_j_output) - len(R_j_input)) - len(R_j_1) * (
                len((D_i_output & R_j_1) | (D_i_1 & R_j_output)) - len(D_i_input & R_j_input))
        if i > 0:
            newPr_tag = 1
        elif i < 0:
            newPr_tag = -1
        else:
            newPr_tag = 0
        return newPr_tag


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
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
skf = StratifiedKFold(n_splits=K)
results_X_1 = {}
results_y_1 = {}
results_X_input = {}
results_y_input = {}
results_1 = {}
LEN = {}
F1 = {}
results_X = group_indices_by_features(X)
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if fold == 0:
        results_y_1 = group_indices(y_train)
        results_X_1 = group_indices_by_features(X_train)
        results_y_input = group_indices(y_test)
        results_X_input = group_indices_by_features(X_test)
        for D_i_key in results_y_1.keys():
            results_1[D_i_key] = {}
            POS = []
            BND = []
            NEG = []
            for R_j_key in results_X_1.keys():
                probability, intersection_length = calProbability(results_y_1.get(D_i_key), results_X_1.get(R_j_key))
                LEN[f"{D_i_key}-{R_j_key}"] = intersection_length
                divThreeArea(probability, a, b, R_j_key, POS, BND, NEG)
            results_1[D_i_key]["POS"] = POS
            results_1[D_i_key]["BND"] = BND
            results_1[D_i_key]["NEG"] = NEG
            y_pred = predict(X_test, POS, BND, NEG, D_i_key)
            F1[f"{fold}--{D_i_key}"] = calculate_f1_score(y_test, y_pred, D_i_key)
    else:
        results = {}
        results_y_l = {}
        results_X_l = {}
        results_y_output = group_indices(y_test)
        results_X_output = group_indices_by_features(X_test)
        results_y_l = calTSet(results_y_1, results_y_output, results_y_input)
        results_X_l = calCSet(results_X_1, results_X_output, results_X_input)
        for D_i_key in results_y_l.keys():
            D_i_1 = results_y_1.get(D_i_key, set())
            D_i_output = results_y_output.get(D_i_key, set())
            D_i_input = results_y_input.get(D_i_key, set())
            result = results_1[D_i_key]
            P = result.get("POS", [])
            B = result.get("BND", [])
            N = result.get("NEG", [])
            newP = []
            newB = []
            newN = []
            results_X_l_keys = set(results_X_l)
            for P_element in P:
                if P_element in results_X_l_keys:
                    results_X_l_keys.remove(P_element)
                    newPr_tag = calNewProbability(D_i_1, D_i_output, D_i_input, results_X_1.get(P_element, set()),
                                                  results_X_output.get(P_element, set()),
                                                  results_X_input.get(P_element, set()),
                                                  LEN.get(f"{D_i_key}-{P_element}", 0))
                    if newPr_tag == 0 or newPr_tag == 1:
                        newP.append(P_element)
                    else:
                        newPr = len(results_y_l.get(D_i_key) & results_X_l.get(P_element)) / len(results_X_l.get(P_element))
                        divThreeArea(newPr, a, b, P_element, newP, newB, newN)
            for B_element in B:
                if B_element in results_X_l_keys:
                    results_X_l_keys.remove(B_element)
                    newPr_tag = calNewProbability(D_i_1, D_i_output, D_i_input, results_X_1.get(B_element, set()),
                                                  results_X_output.get(B_element, set()),
                                                  results_X_input.get(B_element, set()),
                                                  LEN.get(f"{D_i_key}-{B_element}", 0))
                    if newPr_tag == 0:
                        newB.append(B_element)
                    else:
                        newPr = len(results_y_l.get(D_i_key) & results_X_l.get(B_element)) / len(results_X_l.get(B_element))
                        divThreeArea(newPr, a, b, B_element, newP, newB, newN)
            for N_element in N:
                if N_element in results_X_l_keys:
                    results_X_l_keys.remove(N_element)
                    newPr_tag = calNewProbability(D_i_1, D_i_output, D_i_input, results_X_1.get(N_element, set()),
                                                  results_X_output.get(N_element, set()),
                                                  results_X_input.get(N_element, set()),
                                                  LEN.get(f"{D_i_key}-{N_element}", 0))
                    if newPr_tag == -1 or newPr_tag == 0:
                        newN.append(N_element)
                    else:
                        newPr = len(results_y_l.get(D_i_key) & results_X_l.get(N_element)) / len(results_X_l.get(N_element))
                        divThreeArea(newPr, a, b, N_element, newP, newB, newN)
            for element in results_X_l_keys:
                newPr = len(results_y_l.get(D_i_key) & results_X_l.get(element)) / len(results_X_l.get(element))
                divThreeArea(newPr, a, b, element, newP, newB, newN)
            y_pred = predict(X_test, newP, newB, newN, D_i_key)
            F1[f"{fold}--{D_i_key}"] = calculate_f1_score(y_test, y_pred, D_i_key)

F1_sum = sum(F1.values())
F1_avg = F1_sum / len(F1)
end = time.time()
print("count(R)=",len(results_X.keys()))
print("F1:", F1)
print("F1_avg=",F1_avg)
print("time:", end - start)