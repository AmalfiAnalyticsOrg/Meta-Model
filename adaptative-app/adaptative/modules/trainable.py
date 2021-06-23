from soil import decorator, task, modulify, task_wait
from soil.data_structures.data_structure import DataStructure
from soil.data_structures.sklearn_data_structure import SKLearnDataStructure
from sklearn.model_selection import cross_val_score
from soil import logger
import time
import pandas as pd


@decorator(depth=3)
def trainable(module):
    def decorator_1(**model_params):
        def train_fn(**train_params):
            def inner_module(*data):
                return train(*data, module=module, model_params=model_params, **train_params)
            return inner_module
        return train_fn
    return decorator_1


@modulify(output_types=lambda *input_types, **args: [DataStructure])
def train(*data, module=None, model_params=None, **kwargs):
    logger.info('train {} kwargs: {}'.format(module, kwargs))
    res = task(module)(*data, **model_params)
    res, = task_wait(res)
    return [res]
