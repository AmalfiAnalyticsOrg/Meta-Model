''' Module to train the predictor '''
import pandas as pd

from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.trees import HoeffdingTreeClassifier
# from soil.data_structures.sklearn_data_structure import SKLearnDataStructure, Model_DT, Model_RF
from soil.data_structures.sklearn_data_structure import SKLearnDataStructure
from soil.modules.trainable import trainable
from soil import modulify
from soil import logger
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from time import sleep
import copy
from sklearn.preprocessing import LabelEncoder


@trainable
@modulify(output_types=lambda *input_types, **args: [SKLearnDataStructure])
def train_base_models(data, constructor=None, model_params=None, **train_params):
    ''' Trains the predictor with available data '''

    # Getting the model hyper-parameters
    if model_params is None:
        model_params = {}

    # Instanciating the model
    model = constructor(**model_params)
    metadata = data.metadata

    #  This is here to prevent a weird bug.
    if data.data is not None:
        data = data.data

    # Getting the name of the variables we want to use
    x_labels = metadata['columns']['X_columns']
    y_labels = metadata['columns']['y_columns']


    df = pd.DataFrame(data)
    x = df[x_labels]
    y = df[y_labels].values.ravel()
    trans = {}
    inv = {}
    if constructor == HoeffdingTreeClassifier:
        x = x.to_numpy()
        for i, k in enumerate(set(y)):
            trans[k] = i
            inv[i] = k

        y = [trans[i] for i in y]

    logger.info('Training %s', model)
    model = model.fit(x, y)

    if constructor == HoeffdingTreeClassifier:
        accuracy = model.score(x, y)

    else:
        accuracy = cross_val_score(model, x, y, cv=10)
        accuracy = sum(accuracy)/len(accuracy)

    # Metadata
    model_metadata = {
        'x_labels': x_labels,
        'y_labels': y_labels,
        'time': time.time(),
        'accuracy': round(accuracy, 2),
        'model_type': 'sklearn',
        'constructor': str(constructor.__name__),
        'model_hyperparameters': model_params,
        'id': train_params['id'],
        'transformation': trans,
        'invers_trans': inv,
        }

    # In case we need some specific get data
    # name = constructor.__name__
    # dic = {'RandomForestClassifier': Model_DT,
    #        'DecisionTreeClassifier': Model_RF}

    # return [dic[name](model, model_metadata)]
    # sleep(30)

    return [SKLearnDataStructure(model, model_metadata)]
