import unittest
import logging
from soil.modules.static.train_base_models import train_base_models

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skmultiflow.trees import HoeffdingTreeClassifier
from soil.modules.static.train_MM import assembling, training
from soil.modules.static.MM_methods import SVM_weights, accuracy_weights
import soil
import time
from soil.modules.make_predictions import make_predictions
import pandas as pd
import random


class TestTrain(unittest.TestCase):

    def test_train_1(self):
        data = soil.data('iris')
        training_params = {'method': accuracy_weights}
        config_models = [(RandomForestClassifier, {"n_estimators": 12, "random_state": random.seed(1234)}),
                         (DecisionTreeClassifier, {"random_state": random.seed(1234)})]

        models = list([train_base_models(constructor=constructor, model_params=model_params, **{'id': i})
                      for i, (constructor, model_params) in enumerate(config_models)])

        predictor_ref, = assembling(models=models, **training_params)()(data)

        predictor_ref, = training(data, predictor_ref, **training_params)
        print(predictor_ref.metadata)
        self.assertEqual(predictor_ref.metadata['trained'], True, "Error Msg")


def filterKeys(document, use_these_keys):
    return {key: document[key] for key in use_these_keys
            if str(document[key]) not in {"empty", 'nan', 'NaT', 'T', 'SC'}}


def rows_generator(df):
    rows = []
    for _, document in df.iterrows():
        source_ = filterKeys(document, df.columns)
        rows.append(source_)
    return rows


class TestPredict(unittest.TestCase, unlabeled_data_file=None):

    def test_train_1(self, unlabeled_data_file=None):
        # Get the model
        model = soil.data('laura_pred')

        # Read data
        data = pd.read_csv(unlabeled_data_file)
        data = rows_generator(data)
        data_ref = soil.data(data, metadata={})

        # Make predictions
        preds, = make_predictions(model, data_ref)
        soil.alias('preds', preds)
        pred_true = ['versicolor',
                     'setosa',
                     'versicolor',
                     'setosa',
                     'versicolor',
                     'virginica',
                     'versicolor',
                     'versicolor',
                     'versicolor',
                     'versicolor']
        pred = [preds.data[i]['target'] for i in range(len(preds))]
        self.assertEqual(pred, pred_true, "Error Msg")
        # assert pred.all() == pred_true.all()


if __name__ == '__main__':

    unittest.main()

# >>> python testTest.py