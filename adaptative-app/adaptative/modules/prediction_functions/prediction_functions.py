import numpy as np
from collections import Counter
from soil import modulify, logger
from soil.data_structures.sklearn_data_structure import MetaModel


@modulify(output_types=lambda *input_types, **args: [MetaModel])
def weighted_majority_bin(trained_models, *data_inputs, **train_params):
    '''
    Weighted majority for binary classifiers
    '''

    w = 1/len(trained_models) * np.ones(len(trained_models))

    # This is here to prevent a weird bug.
    if data_inputs[0].data is not None:
        data = data_inputs[0].data

    for row in data:
        predictions = []
        p = np.dot(w, predictions)
        y_hat = np.sign(p - train_params['threshold'])
        row['mm'] = y_hat
        for i in range(len(trained_models)):
            if predictions[i] != row['y']:  # row[y], y true for the observation x
                w[i] = train_params['beta'] * w[i]
        s = np.sum(w)
        w = [w[i]/s for i in range(len(trained_models))]

    return w


def weighted_majority(Y, weight):
    '''
    Weighted majority for classifiers
    Y: matrix (prediccions x models)
    weight: weights of each model
    '''

    preds = []

    # For each prediction
    for i in range(len(Y[:, 0])):
        predictions = [Y[i][j] for j in range(len(weight))]
        freqs = Counter(predictions)  # dictionary key = predicted class, value = frequency
        freqs2 = {k: 0 for k in freqs.keys()}
        for j in range(len(weight)):
            freqs2[Y[i][j]] += freqs[Y[i][j]] * weight[j]

        # We take the key with the maximum value
        pred = max(freqs2.items(), key=lambda x: x[1])[0]

        preds.append(pred)

    return preds
