import pandas as pd
import numpy as np
from sklearn import svm
from soil import logger
from soil import modulify, task
from soil.modules.make_predictions import make_predictions
from soil.data_structures.predefined.list import List


@modulify(output_types=lambda *input_types, **args: [List])
def get_SVM_weights(preds, data):
    # preds = list of references
    metadata = data.metadata
    # #  This is here to prevent a weird bug.
    if data.data is not None:
        data = data.data

    y_labels = metadata['columns']['y_columns']

    df = pd.DataFrame(data)
    y = df[y_labels]

    df_preds = pd.DataFrame(preds.data)
    X = df_preds[y_labels]
    logger.info('This is X')
    logger.info(X)

    SVM = svm.SVC()
    SVM = SVM.fit(X, y)
    weights = SVM.class_weight()
    weights = []

    return [List(weights, metadata=[])]


def SVM_weights(data, models):

    preds = []
    for mod in models:
        pred, = task(make_predictions)(mod, data)
        logger.info("pred")
        logger.info(pred)
        preds.append(pred)

    weights, = task(get_SVM_weights)(preds[0], data)

    return weights.data


def accuracy_weights(data, models):

    weights = [model.metadata['accuracy'] for model in models]
    weights = weights/np.sum(weights)

    return weights.tolist()
