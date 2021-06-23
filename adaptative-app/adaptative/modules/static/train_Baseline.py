''' Module to train the predictor '''
import pandas as pd
import time
from soil.data_structures.baseline_model import BaselineModel
from soil.modules.trainable import trainable
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from soil import modulify, logger
from soul.modules.static.baselines_class import Mean_baseline

@trainable
@modulify(output_types=lambda *input_types, **args: [BaselineModel])
def train_DT(data, model_params=None, **train_params):
    ''' Trains the predictor with available data '''
    if model_params is None:
        model_params = {}
    model = Mean_baseline(**model_params)
    metadata = data.metadata

    # #  This is here to prevent a weird bug.
    if data.data is not None:
        data = data.data

    x_labels = metadata['columns']['X_columns']
    y_labels = metadata['columns']['y_columns']

    df = pd.DataFrame(data)
    x = df[x_labels]
    y = df[y_labels].values.ravel()

    model = model.fit(data)

    accuracy = cross_val_score(model, x, y, cv=10)
    accuracy = sum(accuracy)/len(accuracy)

    model_metadata = {
        'x_labels': x_labels,
        'y_labels': y_labels,
        'time': time.time(),
        'accuracy': accuracy,
        }

    return [BaselineModel(model, model_metadata)]
