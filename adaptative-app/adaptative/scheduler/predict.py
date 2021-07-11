''' Module to predict with a trained predictor. '''
import argparse
import logging
import soil
from soil import logger
from soil.modules.make_predictions import make_predictions
import pandas as pd
# from soil.modules.prediction_functions.prediction_function_weighted_majority import weighted_majority

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def filterKeys(document, use_these_keys):
    return {key: document[key] for key in use_these_keys
            if str(document[key]) not in {"empty", 'nan', 'NaT', 'T', 'SC'}}


def rows_generator(df):
    rows = []
    for _, document in df.iterrows():
        source_ = filterKeys(document, df.columns)
        rows.append(source_)
    return rows


def get_predictions(unlabeled_data_file=None):
    ''' Main function generate preds. '''

    # Get the model
    model = soil.data('laura_pred')
    print(model.metadata)

    # Read data
    data = pd.read_csv(unlabeled_data_file)
    data = rows_generator(data)
    data_ref = soil.data(data, metadata={})

    # Make predictions
    # res, = to_es_data_structure(data_ref) # redundant només si vull guardar
    preds, = make_predictions(model, data_ref)
    soil.alias('preds', preds)
    logger.info(preds.data)


def main():
    ''' Argument parsing. '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--unlabeled-data-file', type=argparse.FileType('r'),
        help='Pass unlabeled data here', required=True
    )
    args = parser.parse_args()
    get_predictions(**vars(args))


if __name__ == '__main__':
    main()
