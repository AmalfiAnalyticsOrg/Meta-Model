from soil import modulify, task, task_wait
from soil.data_structures.sklearn_data_structure import MetaModel
from soil import logger
from soil.modules.trainable import trainable
import time
from soil.data_structures.predefined.list import List
import numpy as np
from time import sleep
import sys


@trainable
@modulify(output_types=lambda *input_types, **args: [MetaModel])
def _train(data, models=None, **train_params):

    metadata = data.metadata

    x_labels = metadata['columns']['X_columns']
    y_labels = metadata['columns']['y_columns']

    if data.data is not None:
        data = data.data

    data = list(data)
    data = List(data, metadata=metadata)

    # Trained models
    trained_models = []

    # Training models
    for model in models:
        logger.info('This is the model: ')
        logger.info(model)
        # assert model.metadata['model_type'] in ['amalfi-metamodel', 'sklearn'], "model not implemented"
        try:
            # passing here or in train.py
            mod, = task(model())(data)
            trained_models.append(mod)
        except ValueError:
            logger.info(ValueError)
            logger.error('NOT implemented model; task try')
            logger.info(model)
    trained_models = task_wait(trained_models)

    for mod in trained_models:
        if 'constructor' in mod.metadata.keys():
            logger.info('I have constructor')

        if 'model_hyperparameters' in mod.metadata.keys():
            logger.info('I have hyp')

        else:
            logger.info('Oh, i dont')
            logger.info(mod.metadata)

    constructors = list([trained_models[i].metadata['constructor'] for i in range(len(trained_models))])
    hyperparams = list([trained_models[i].metadata['model_hyperparameters'] for i in range(len(trained_models))])
    logger.info('trained_models')
    logger.info(trained_models)
    # MetaModel weights
    # weights = [model.metadata['accuracy'] for model in trained_models]
    weights = train_params['method'](data, trained_models)

    logger.info('train params')
    logger.info(train_params['method'])

    # MetaModel data
    model_metadata = {
        'x_labels': x_labels,
        'y_labels': y_labels,
        'time': time.time(),
        'weight': weights,
        'model_type': 'amalfi-metamodel',
        'accuracy': round(np.mean([model.metadata['accuracy'] for model in trained_models]), 2),
        'composed_by': constructors,
        'model_hyperparameters': hyperparams,
        'constructor': 'metamodel'
        }

    return [MetaModel(trained_models, model_metadata)]


@trainable
@modulify(output_types=lambda *input_types, **args: [MetaModel])
def assembling(data, models=None, **train_params):
    
    # Get the metadata from data
    metadata = data.metadata

    # Get the actual data
    if data.data is not None:
        data = data.data

    # Converting data to List DS
    data = list(data)
    data = List(data, metadata=metadata)

    # Trained models
    trained_models = []

    # Training models
    for model in models:
        try:
            # Tha data can be passed here or in the train.py
            mod, = task(model())(data)
            trained_models.append(mod)
        except ValueError:
            logger.info(ValueError)
            logger.error('NOT implemented model; task try')
            logger.info(model)

    # Syncronization barrier to ensure all base models have been trained
    trained_models = task_wait(trained_models)
    for model in trained_models:
        logger.info(model.metadata)
    
    metadata = {
        'Assembled': True,
    }

    return [MetaModel(trained_models, metadata)]


@modulify(output_types=lambda *input_types, **args: [MetaModel])
def training(data, models, **train_params):
    
    logger.info("Hi there")
    
    metamodel_metadata = models.metadata
    data_metadata = data.metadata

    x_labels = data_metadata['columns']['X_columns']
    y_labels = data_metadata['columns']['y_columns']

    # Get the actual data avoiding a weird bug
    if data.data is not None:
        data = data.data

    # Set it so other modules can use it
    data = List(data, metadata=data_metadata)

    # Get the base models avoiding a weird bug
    if models.data is not None:
        models = models.data

    # Getting base models metadata
    constructors = []
    hyperparams = []
    accuracy = []
    for model in models:
        model_metadata = model.metadata
        constructors.append(model_metadata['constructor'])
        constructors.append(model_metadata['model_hyperparameters'])
        accuracy.append(model_metadata['accuracy'])

    # MetaModel weights
    weights = train_params['method'](data, models)


    # MetaModel data
    new_metamodel_metadata = {
        'x_labels': x_labels,
        'y_labels': y_labels,
        'time': time.time(),
        'weight': weights,
        'model_type': 'amalfi-metamodel',
        'accuracy': round(np.mean(accuracy), 2),
        'composed_by': constructors,
        'model_hyperparameters': hyperparams,
        'constructor': 'metamodel',
        'trained': True,
        'id': 9999
    }

    return [MetaModel(models, new_metamodel_metadata)]


@modulify(output_types=lambda *input_types, **args: [MetaModel])
def return_data(pred):
    if pred.data is not None:
        pred = pred.data
    logger.info('im the printeeerr')
    logger.info(pred)
    metadata = {}
    return[MetaModel(pred, metadata=metadata)]