''' Module to train the predictor '''
import pandas as pd
import time
from soil.data_structures.historcial_model import HistoricalModel
from soil.modules.trainable import trainable
from soil import modulify
from soil import logger
from sklearn.model_selection import cross_val_score


@trainable
@modulify(output_types=lambda *input_types, **args: [HistoricalModel])
def train_Historical(data, model_params=None, **train_params):
    ''' Trains the predictor with available data '''
    if model_params is None:
        model_params = {}
    metadata = data.metadata

    # #  This is here to prevent a weird bug.
    if data.data is not None:
        data = data.data

    x_labels = metadata['columns']['X_columns']
    y_labels = metadata['columns']['y_columns']

    df = pd.DataFrame(data)
    x = df[x_labels]
    y = df[y_labels].values.ravel()

    model = []

    accuracy = cross_val_score(model, x, y, cv=10)
    accuracy = sum(accuracy)/len(accuracy)

    logger.info('Model oob score: %s', model.oob_score_)
    logger.info('Model error: %s', 1 - model.oob_score_)

    model_metadata = {
        'oob_score': model.oob_score_,
        'error': 1 - model.oob_score_,
        'x_labels': x_labels,
        'y_labels': y_labels,
        'time': time.time(),
        'accuracy': accuracy
        }

    return [HistoricalModel(model, model_metadata)]
