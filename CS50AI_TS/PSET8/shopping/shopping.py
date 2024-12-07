import csv
import sys
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif

TEST_SIZE = 0.4


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    evidence, labels = load_data(sys.argv[1])
    evidence = handle_missing_data(evidence)
    evidence = standardize_data(evidence)

    X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=TEST_SIZE)

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    sensitivity, specificity = evaluate(y_test, predictions)
    evaluate_metrics(y_test, predictions)

    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

    cross_validate_model(model, evidence, labels)


def load_data(filename):
    evidence = []
    labels = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_num = enumerate(months)
    month = {k: v for v, k in month_num}

    with open(filename, "r") as raw:
        reader = csv.DictReader(raw)
        for each_row in reader:
            row = []
            row.append(int(each_row["Administrative"]))
            row.append(float(each_row["Administrative_Duration"]))
            row.append(int(each_row["Informational"]))
            row.append(float(each_row["Informational_Duration"]))
            row.append(int(each_row["ProductRelated"]))
            row.append(float(each_row["ProductRelated_Duration"]))
            row.append(float(each_row["BounceRates"]))
            row.append(float(each_row["ExitRates"]))
            row.append(float(each_row["PageValues"]))
            row.append(float(each_row["SpecialDay"]))
            row.append(int(month[each_row["Month"]]))
            row.append(int(each_row["OperatingSystems"]))
            row.append(int(each_row["Browser"]))
            row.append(int(each_row["Region"]))
            row.append(int(each_row["TrafficType"]))
            row.append(int(each_row["VisitorType"] == 'Returning_Visitor'))
            row.append(int(each_row["Weekend"] == 'TRUE'))
            evidence.append(row)
            labels.append(int(each_row["Revenue"] == 'TRUE'))

    return evidence, labels


def handle_missing_data(evidence):
    evidence = np.array(evidence)
    for i in range(evidence.shape[1]):
        col = evidence[:, i]
        col_mean = np.nanmean(col)
        col[np.isnan(col)] = col_mean
    return evidence.tolist()


def standardize_data(evidence):
    scaler = StandardScaler()
    return scaler.fit_transform(evidence)


def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    size = len(labels)
    negatives = 0
    positives = 0
    true_positives = 0
    true_negatives = 0

    for i in range(size):
        if labels[i] == 0:
            negatives += 1
            if labels[i] == predictions[i]:
                true_negatives += 1
        else:
            positives += 1
            if labels[i] == predictions[i]:
                true_positives += 1

    sensitivity = true_positives / positives
    specificity = true_negatives / negatives
    return sensitivity, specificity


def evaluate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


def cross_validate_model(model, evidence, labels):
    scores = cross_val_score(model, evidence, labels, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean score: {scores.mean()}")


if __name__ == "__main__":
    main()
