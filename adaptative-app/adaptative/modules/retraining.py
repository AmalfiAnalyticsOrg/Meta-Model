import soil
import numpy as np
from soil import logger
from soil.modules.static.train_SKLearn import train_SKLearn
from soil.data_structures.sklearn_data_structure import MetaModel
from soil.data_structures.baseline_model import IDS
from soil import modulify
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time

# from soil.data_structures.predefined.list import List


# @modulify(output_types=lambda *input_types, **args: [MetaModel])
# def retraining(data, model, adaptation_fun=None):
#     logger.info('Ready to retrain')

#     cons = {
#         # 'RandomForestClassifier': RandomForestClassifier,
#         'RandomForestClassifier': RandomForestClassifier,
#         'DecisionTreeClassifier': DecisionTreeClassifier
#     }
#     logger.info('HERE!')
#     if model.data is not None:
#         small_models = model.data
    
#     logger.info('small_models')
#     logger.info(small_models.data)

#     to_consider = np.zeros(len(small_models.data))
#     for i, model in enumerate(small_models.data):
#         # KickOut ha de ser adaptation function no un string amb ifs
#         if adaptation_fun == 'KickOutOlder':
#             to_consider[i] = model.metadata['time']
#         elif adaptation_fun == 'KickOutWorst':
#             to_consider[i] = model.metadata['accuracy']

#     older = np.argmin(to_consider)
#     # Guardar a metadata el constructor + hyperparams si vull mantenir mateixa policy
#     # depèn del q vulgui mantinc freq. de models simples o no
#     if model.metadata['model_type'] == 'sklearn':

#         logger.info('model.metadata[constructor]')
#         logger.info(model.metadata['constructor'])

#         constructor = cons[str(model.metadata['constructor'])]
#         logger.info(constructor)
#         hyperparams = model.metadata['model_hyperparameters']
#         logger.info('hyperparams')
#         logger.info(hyperparams)
#         hyperparams = {'n_estimators': 12}
#         mod = train_SKLearn(constructor=RandomForestClassifier, model_params={})()(data)
#         small_models[older] = train_SKLearn(constructor=RandomForestClassifier, model_params={})()(data)

#     metadata = {}
#     return [MetaModel(small_models, metadata)]


@modulify(output_types=lambda *input_types, **args: [MetaModel])
def retraining(data, model, adaptation_fun=None):
    logger.info('Ready to retrain')

    logger.info('HERE!')
    if model.data is not None:
        small_models = model.data

    logger.info('small_models')
    logger.info(small_models.data)

    to_consider = np.zeros(len(small_models.data))
    for i, model in enumerate(small_models.data):
        # KickOut ha de ser adaptation function no un string amb ifs
        # Menyeee!
        if adaptation_fun == 'KickOutOlder':
            to_consider[i] = model.metadata['time']
        elif adaptation_fun == 'KickOutWorst':
            to_consider[i] = model.metadata['accuracy']

    # older = np.argmin(to_consider)

    # logger.info('small_models[older]')
    # logger.info(small_models.data[older])

    # if small_models.data[older].metadata['model_type'] == 'sklearn':
    #     sklearn = True
    # else:
    #     sklearn = False

    constructor = small_models.data[older].metadata['constructor']
    model_params = small_models.data[older].metadata['model_hyperparameters']
    # # Guardar a metadata el constructor + hyperparams si vull mantenir mateixa policy
    # # depèn del q vulgui mantinc freq. de models simples o no
    # if model.metadata['model_type'] == 'sklearn':

    #     logger.info('model.metadata[constructor]')
    #     logger.info(model.metadata['constructor'])

    #     constructor = cons[str(model.metadata['constructor'])]
    #     logger.info(constructor)
    #     hyperparams = model.metadata['model_hyperparameters']
    #     logger.info('hyperparams')
    #     logger.info(hyperparams)
    #     hyperparams = {'n_estimators': 12}
    #     mod = train_SKLearn(constructor=RandomForestClassifier, model_params={})()(data)
    #     small_models[older] = train_SKLearn(constructor=RandomForestClassifier, model_params={})()(data)

    metadata = {
        # 'to_modify': int(older),
        # 'sklearn': sklearn,
        # 'constructor': constructor,
        # 'model_hyperparameters': model_params,
        'retrained': True
    }

    return [MetaModel([], metadata)]


@modulify(output_types=lambda *input_types, **args: [MetaModel])
def addModel(metamodel, new_model):
    logger.info('wwwooo there!')
    
    metadata = metamodel.metadata
    if metamodel is not None:
        metamodel = metamodel.data


    metamodel.append(new_model)
    new_metadata = metadata
    new_metadata['time'] =int(time.time())
    new_metadata["oh my gah!"] = "Azu"

    return [MetaModel(metamodel, metadata)]


@modulify(output_types=lambda *input_types, **args: [MetaModel])
def removeModel(models, worst_id=None):

    metamodel_metadata = models.metadata
    if models is not None:
        models = models.data

    remove_id = worst_id['id']
    logger.info(remove_id)
    new_metamodel = []
    # for i, model in enumerate(models):
    #     # Aqui hi ha un problema logic amb aixo dels ids.   
    #     # No troba cap amb aquell id. Weird eh!
    #     if i != 0:
    #         new_metamodel.append(model)

    logger.info('len(models)')
    logger.info(len(models))

    for model in models:
        # Aqui hi ha un problema logic amb aixo dels ids.   
        # No troba cap amb aquell id. Weird eh!
        if model.metadata['id'] != remove_id:
            new_metamodel.append(model)

    logger.info('len(new_metamodel)')
    logger.info(len(new_metamodel))

    new_metamodel_metadata = metamodel_metadata
    new_metamodel_metadata["hm"]= "la laura mola"

    return [MetaModel(new_metamodel, new_metamodel_metadata)]


@modulify(output_types=lambda *input_types, **args: [MetaModel])
def getWorst(models):
    
    metamodel_metadata = models.metadata
    
    if models.data is not None:
        models = models.data
    

    acc = np.zeros(len(models))
    timestamp = np.zeros(len(models))

    for i, model in enumerate(models):
        model_metadata = model.metadata
        acc[i] = model_metadata['accuracy']
        timestamp[i] = model_metadata['time']

    # Where is the worst?
    to_remove = np.argmin(acc)
    id_time = timestamp[to_remove]
    
    # Who is it?
    worst_metadata = models[to_remove].metadata
    model_type = worst_metadata['model_type']
    hyper = worst_metadata['model_hyperparameters']
    cons = worst_metadata['constructor']
    worst_id = worst_metadata['id']

    new_worst_metadata = {
        'id': worst_id,
        'model_type': model_type,
        'model_hyperparameters': hyper,
        'constructor': cons
    }
    # new_worst_metadata = {}
    return [MetaModel([], new_worst_metadata)]


@modulify(output_types=lambda *input_types, **args: [MetaModel])
def getOlder(models):
    
    metamodel_metadata = models.metadata
    
    if models.data is not None:
        models = models.data

    timestamp = np.zeros(len(models))

    for i, model in enumerate(models):
        model_metadata = model.metadata
        timestamp[i] = model_metadata['time']
        logger.info('TIME')
        logger.info(i)
        logger.info('timestamp_i')
        logger.info(timestamp[i])

    # Where is the older?
    to_remove = np.argmin(timestamp)
    
    # Who is it?
    older_metadata = models[to_remove].metadata
    model_type = older_metadata['model_type']
    hyper = older_metadata['model_hyperparameters']
    cons = older_metadata['constructor']
    older_id = older_metadata['id']

    new_older_metadata = {
        'id': older_id,
        'model_type': model_type,
        'model_hyperparameters': hyper,
        'constructor': cons
    }

    return [MetaModel([], new_older_metadata)]


@modulify(output_types=lambda *input_types, **args: [MetaModel])
def reavaluate_single_models(model, data):
    model_metadata = model.metadata
    if model.data is not None:
        model = model.data
    
    data_metadata = data.metadata
    if data.data is not None:
        data = data.data

    reevaluated = []
    # Let's evaluate the performance of each small model
    for mod in model:
        new_accuracy = 0
        mod.metadata['accuracy'] = new_accuracy
        reevaluated.append(mod)


@modulify(output_types=lambda *input_types, **args: [IDS]) # LIST
def getting_ids(model):
    if model.data is not None:
        model = model.data

    model_ids = []
    for mod in model:
        model_ids.append(mod.metadata['id'])

    return[IDS(model_ids, {'ids': model_ids})]


@modulify(output_types=lambda *input_types, **args: [MetaModel])
def taking_small(model, _id=None):
    if model.data is not None:
        model = model.data

    small_model = []
    metadata = {'no metadata': True}
    logger.info('IM TAKING THE SMALLLLL')

    for mod in model:
        if mod.metadata['id'] == _id:
            small_model.append(mod)
            metadata = mod.metadata
            metadata['no metadata'] = False
            logger.info('metadata')
            logger.info(metadata)

    if len(small_model):
        model_to_use = small_model[0].data
    
    else:
        model_to_use = []

    return[MetaModel(model_to_use, metadata)]
