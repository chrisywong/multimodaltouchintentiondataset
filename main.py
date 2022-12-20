import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import neighbors
import sklearn.model_selection as sklmodelsel
from sklearn.metrics import confusion_matrix
import time
from joblib import dump, load


def create_model(raw_data: str, model="knn11", cross_validation=False, outputflag=False, outputname="outputmodel"):
    input_dataset_name = raw_data
    output_model_name = "models\\" + outputname + ".joblib"

    # CSV READER
    with open(input_dataset_name + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        matrix_data = []
        for row in reader:
            row_data = []
            for el in row:
                row_data.append(float(el))
            matrix_data.append(row_data)

    # DATA INITIALISATION
    np.random.seed(0)
    np.random.shuffle(matrix_data)
    y = np.array([matrix_data[k][-1] for k in range(len(matrix_data))]) # ground truth
    x = np.array([matrix_data[k][0:-1] for k in range(len(matrix_data))]).reshape(-1, len(matrix_data[0]) - 1) # reduced feature set

    if model == "nn":
        model_sklearn = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,), random_state=1, max_iter=1000)
    elif model == "svm":
        model_sklearn = svm.SVC(gamma='auto')
    elif model == "knn11":
        model_sklearn = neighbors.KNeighborsClassifier(n_neighbors=11, weights='distance')
    elif model == "knn51":
        model_sklearn = neighbors.KNeighborsClassifier(n_neighbors=51, weights='distance')
    else:
        raise AssertionError("model {} not known".format(model))

    # MODEL TRAINING
    if cross_validation:
        t_start = time.time()
        scores = sklmodelsel.cross_validate(model_sklearn, x, y, cv=5, return_train_score=True, return_estimator=True)
        t_crossval = time.time()
        y_pred = sklmodelsel.cross_val_predict(model_sklearn, x, y, cv=5)
        t_pred = time.time()
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        totval = tp + fp + tn + fn

        print('===================')
        print("Using ", model)
        print("[X-Valid] Average train", np.average(scores['train_score']), ' // Average test',
              np.average(scores['test_score']))
        print("[X-Valid] Confusion Matrix:")
        print("\t tp \t tn \t fp \t fn")
        print("\t", tp, "\t", tn, "\t", fp, "\t", fn)
        print("\t", round(tp / totval * 100, 2), "\t", round(tn / totval * 100, 2), "\t", round(fp / totval * 100, 2), "\t",
              round(fn / totval * 100, 2))
        print('Time taken crossval:', round(t_crossval - t_start, 3), '[s] / crossval predict:',
              round(t_pred - t_crossval, 3), '[s]')

    print('===================')
    if outputflag:
        print('Saving this model:', model_sklearn)
        model_sklearn.fit(x, y)  # learning
        dump(model_sklearn, output_model_name, protocol=2)
        print('Model successfully saved to', output_model_name)
    else:
        print('Model output false, model not saved')


def model_use(name_model: str, input_data: list):
    assert len(input_data) == 5
    model_sklearn = load("models\\" + name_model)
    y = model_sklearn.predict(np.array(input_data).reshape(1, -1))
    print("Prediction:", y)
    return y


if __name__ == '__main__':
    print("======== Creating model from training data ========")
    dataset_name = 'datasets/1-Dataset_ts_hp_hs_ga_gs'
    create_model(dataset_name, model="knn11", cross_validation=True, outputflag=False, outputname="outputmodel")

    print("======== Predicting using trained model ========")
    x = [1,	0.04539232395709132, 0.5595150281618438, 0.5986402764736726, 0.07003290615123266]
    model_use("knn11_1.joblib", x)

