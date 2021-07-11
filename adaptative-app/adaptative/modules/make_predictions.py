''' Module to do the predictions given a model and data points to predict on. '''
from soil import modulify, task, task_wait
from soil.data_structures.predictions import Predictions
import pandas as pd
from soil import logger
import numpy as np
import copy
from soil.modules.prediction_functions.prediction_functions import weighted_majority
from soil.data_structures.predefined.list import List


@modulify(output_types=lambda *input_types, **args: [Predictions])
def make_predictions(models, data):

    ''' Module function to do the predictions given a model and data points to predict on. '''

    models_metadata = models.metadata

    data_metadata = data.metadata
    if data.data is not None:
        data = data.data

    data_to_models = List(data, data_metadata)
    if models.data is not None:
        models = models.data

    df = pd.DataFrame(data)

    if models_metadata['model_type'] == 'sklearn':

        x_labels = models_metadata['x_labels']
        X = df[x_labels]
        if models_metadata['constructor'] == 'HoeffdingTreeClassifier':
            X = X.to_numpy()

        y = models.predict(X)  # -> sklearn

    elif models_metadata['model_type'] == 'amalfi-metamodel':
        logger.info('JEJEJE')
        Y = np.zeros([df.shape[0], len(models)]).astype(object)
        for i, model in enumerate(models):
            # To check
            model_metadata = model.metadata
            if model_metadata['model_type'] == 'amalfi-metamodel':
                recursive = task(make_predictions)(model, data_to_models)
                recursive = task_wait(recursive)
            else:
                _model = model.data
                x_labels = model_metadata['x_labels']
                X = df[x_labels]
                if model.metadata['constructor'] == 'HoeffdingTreeClassifier':
                    X = X.to_numpy()
                local_y = _model.predict(X)
                if model.metadata['constructor'] == 'HoeffdingTreeClassifier':
                    local_y = [model.metadata['invers_trans'][i] for i in local_y]
                Y[:, i] = local_y

        # y = preds_params['method'](Y, weight)
        weight = models_metadata['weight']
        y = weighted_majority(Y, weight)

    else:
        raise NotImplementedError

    ids = df['id']
    new_df = df[x_labels].copy()
    new_df['target'] = y
    new_df_dict = new_df.to_dict('records')
    predictions = list({'id': ids[i], **v} for i, v in enumerate(new_df_dict))  # -> row_to_actions

    return [Predictions(predictions, metadata={'index': 'roser-simple-preds', 'rewrite': True})]
