''' Module to train a predictor. '''
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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():

    data = soil.data('iris')

    ## Depth One Model ##
    # Parameters we want to use to train the Meta-Model, for example
    # the method which determines which algorithm is used to mix 
    # base models
    training_params = {'method': accuracy_weights}

    # Model configurations, tuples of model constructors plus its parameters
    config_models = [(RandomForestClassifier, {"n_estimators": 12}),
                     (DecisionTreeClassifier, {}),
                     (LinearDiscriminantAnalysis, {}),
                     (HoeffdingTreeClassifier, {})]
   
    # Instancing each base model
    print("Building the models...")
    models = list([train_base_models(constructor=constructor, model_params=model_params, **{'id': i})
                   for i, (constructor, model_params) in enumerate(config_models)])

    # Taining each base model
    print('MODELS', models)
    print('Building the assembly...')
    predictor_ref, = assembling(models=models, **training_params)()(data)

    # Taining the Meta-Model itself
    print("Training the assembly..")
    predictor_ref, = training(data, predictor_ref, **training_params)
    print(predictor_ref.metadata)

    # Saving the predictor with an alias to SOIL
    soil.alias('laura_pred', predictor_ref)

    # Just for testing purposes
    # return predictor_ref.metadata['trained']


if __name__ == '__main__':
    main()
