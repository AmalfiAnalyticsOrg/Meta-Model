''' Module to evaluate predictionss. '''
import argparse
import logging
import soil
import numpy as np

from soil.data_structures.es_data_structure import ESDataStructure
from soil.modules.generate_data_for_training import generate_data_for_training
import pandas as pd
from soil import modulify
from soil import logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# const get_predictions = (predictions, predictive_horizon) = {
#     data = []
#     for pred in predictions: 
#         created = pred.created
#         date = created + predictive_horizon
#         y_pred = pred.data.filter(d => d.date === date )
#         data.append({created, y_pred: y_pred, date: date }
#     return data
# }

# def add_real_value(data, historical):
#     for i in data:
#         i['y_true'] = historical[historical.date == i.date].y


# modulify
def get_predictions(predictions, horizon):
    return 0,


# modulify
def add_real_value(data, historical):
    return 0,


# modulify
def get_predictions_evaluation(data, evulate_fun):
    y_true = data['y']
    y_preds = data['y_preds']
    return evulate_fun(y_true, y_preds)


def evaluate_predictions(evulate_fun, horizon):

    predictions = soil.data('preds')
    historical = soil.data('cmbd')

    data, = get_predictions(predictions, horizon)
    data, = add_real_value(data, historical)

    evaluation, = get_predictions_evaluation(data, evulate_fun)

    print(evaluation)


def main():
    ''' Argument parsing. '''
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    evaluate_predictions(**vars(args))


if __name__ == '__main__':
    main()
