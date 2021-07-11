''' Module to evaluate a predictor. '''
from soil.data_structures.es_data_structure import ESDataStructure
import pandas as pd
from soil import modulify
from soil import logger


def filterKeys(document, use_these_keys):
    return {key: document[key] for key in use_these_keys
            if str(document[key]) not in {"empty", 'nan', 'NaT', 'T', 'SC'}}


def rows_generator(df):
    rows = []
    for _, document in df.iterrows():
        source_ = filterKeys(document, df.columns)
        rows.append(source_)
    return rows


@modulify(output_types=lambda *input_types, **args: [ESDataStructure])
def get_restrospective_data(historical_data, horizon):
    # horizon, day that I recieved data, I want a dataset from the
    # from the horizon until the present (latest data)
    historical_data.data['columns']['timestamp']
    return 0,


@modulify(output_types=lambda *input_types, **args: [ESDataStructure])
def get_random_data(historical_data, test_size=None):
    # TEST SIZE: percentage of the data set that we want to use (0-1)
    #  This is here to prevent a weird bug.
    logger.info('Random data')
    if historical_data.data is not None:
        data = historical_data.data

    df = pd.DataFrame(data)

    df = df.sample(frac=test_size, random_state=1)

    rows = rows_generator(df)

    return [ESDataStructure(rows, metadata=historical_data.metadata)]


def provant_funcions(y_true=None, y_preds=None, evaluate_function=None):
    logger.info('funcio interna')

    r = evaluate_function(y_true, y_preds)
    return r


@modulify(output_types=lambda *input_types, **args: [ESDataStructure])
def get_evaluation(preds, data, evaluate_function=None):
    logger.info('evaluating')

    metadata_true = data.metadata

    if data.data is not None:
        data = data.data

    if preds.data is not None:
        preds = preds.data

    y_labels = metadata_true['columns']['y_columns']

    df_true = pd.DataFrame(data)
    df_true.sort_values(by=['id'], inplace=True)

    df_pred = pd.DataFrame(preds)
    df_pred.sort_values(by=['id'], inplace=True)

    y_true = df_true[y_labels]
    y_pred = df_pred['target']

    results = evaluate_function(y_true, y_pred)

    logger.info('results')
    logger.info(results)

    e = [{evaluate_function.__name__: results}]
    metadata = {'index': 'evaluating', 'rewrite': True}

    return [ESDataStructure(e, metadata=metadata)]
