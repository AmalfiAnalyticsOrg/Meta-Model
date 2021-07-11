''' Module to evaluate the performance of a predictor. '''
import logging
import soil
from soil import logger
from sklearn.metrics import precision_score, accuracy_score
from soil.modules.evaluating_functions_1 import get_restrospective_data, get_evaluation, get_random_data
from soil.modules.make_predictions import make_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def evaluate_model(evaluate_fun, horizon=None, test_size=None):
    historical_data = soil.data('iris')
    model = soil.data('laura_pred')

    if horizon:
        data, = get_restrospective_data(historical_data, horizon)

    if test_size:
        logger.info('getting data')
        data, = get_random_data(historical_data, test_size)

    logger.info('going to evaluate')
    y_pred, = make_predictions(model, historical_data)

    evaluation, = get_evaluation(y_pred, historical_data, evaluate_function=evaluate_fun)

    soil.alias('evaluate', evaluation)
    logger.info(evaluation.data)
    # print(evaluation.data)


def main():
    fun = accuracy_score
    evaluate_model(evaluate_fun=fun)


if __name__ == '__main__':
    main()
