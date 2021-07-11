''' Module to perform an incemental train to a predictor. '''
from soil import logger
import soil
from soil.modules.retraining import retraining, getWorst, removeModel, addModel, getOlder, getting_ids, taking_small
from soil.modules.static.train_MM import training
from soil.modules.static.MM_methods import SVM_weights, accuracy_weights
from soil.modules.static.train_SKLearn import train_SKLearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.metrics import accuracy_score
from soil.modules.make_predictions import make_predictions
from soil.modules.evaluating_functions_1 import get_evaluation


def fun():
    return 



def kickoutWorst(): 

    model = soil.data('laura_pred')
    print("Models's metadata:", model.metadata)

    data = soil.data('iris')
    print("Data's metadata:", data.metadata)

    print("Let's do it!")


    # Option 2: get_data give us the worst id
    print("Let's get the worst guy!")
    worst_one_ref, = getWorst(model)
    worst_one_metadata = worst_one_ref.metadata
    print('im the worst: ', worst_one_metadata)

    # Kick out worst ##
    good_ones, = removeModel(model, worst_id=worst_one_metadata)
    print('retrained at: ', good_ones.metadata)

    # Train new model
    cons = {
        'RandomForestClassifier': RandomForestClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
        'HoeffdingTreeClassifier': HoeffdingTreeClassifier,
    }

    if worst_one_metadata['model_type'] == 'sklearn':
        constructor = cons[worst_one_metadata['constructor']]
        model_params = worst_one_metadata['model_hyperparameters']
        new_model, = train_SKLearn(constructor=constructor, model_params=model_params, **{'id': worst_one_metadata['id']})()(data)

    # reassamble models
    trained_models, = addModel(good_ones, new_model)

    soil.alias('laura_pred', trained_models)


    # modify, = retraining(data, model, adaptation_fun='KickOutOlder')
    # print(modify.metadata)

    # if modify.metadata['sklearn'] == True:
    #     to_modify = modify.metadata['to_modify']
    #     constructor = modify.metadata['constructor']
    #     model_params = modify.metadata['model_hyperparameters']
    
    # cons = {
    #     # 'RandomForestClassifier': RandomForestClassifier,
    #     'RandomForestClassifier': RandomForestClassifier,
    #     'DecisionTreeClassifier': DecisionTreeClassifier
    # }

    # new_model, = train_SKLearn(constructor=cons[constructor], model_params=model_params)()(data)
    
    # print(new_model.metadata)


    # # data = soil.data('iris')
    # # print("Incremental train 2")
    # # training_params = {'method': accuracy_weights}
    # # predictor_ref, = train_2(data, models=trained_models, **training_params)
    # # soil.alias('predictor', predictor_ref)
    # # print(predictor_ref.data)


def kickoutOlder():

    model = soil.data('laura_pred')
    print("Models's metadata:", model.metadata)

    data = soil.data('iris')
    print("Data's metadata:", data.metadata)
    
    print("Let's do it!")

    # Option 2: get_data give us the worst id
    print("Let's get the worst guy!")
    worst_one_ref, = getWorst(model)
    worst_one_metadata = worst_one_ref.metadata
    print('im the worst: ', worst_one_metadata)
    
    # Kick out worst ##
    good_ones, = removeModel(model, worst_id=worst_one_metadata)
    print('retrained at: ', good_ones.metadata)

    # Train new model
    cons = {
        'RandomForestClassifier': RandomForestClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
        'HoeffdingTreeClassifier': HoeffdingTreeClassifier,
    }

    if worst_one_metadata['model_type'] == 'sklearn':
        constructor = cons[worst_one_metadata['constructor']]
        model_params = worst_one_metadata['model_hyperparameters']
        new_model, = train_SKLearn(constructor=constructor, model_params=model_params, **{'id': worst_one_metadata['id']})()(data)

    # reassamble models
    trained_models, = addModel(good_ones, new_model)

    soil.alias('laura_pred', trained_models)


def kickoutOlder():

    model = soil.data('laura_pred')
    print("Models's metadata:", model.metadata)

    data = soil.data('iris')
    print("Data's metadata:", data.metadata)
    
    print("Let's do it!")

    # Option 2: get_data give us the worst id
    print("Let's get the older guy!")
    older_one_ref, = getOlder(model)
    older_one_metadata = older_one_ref.metadata
    print('im the older: ', older_one_metadata)

    # Kick out worst ##
    good_ones, = removeModel(model, worst_id=older_one_metadata)
    print('retrained at: ', good_ones.metadata)

    # Train new model
    cons = {
        'RandomForestClassifier': RandomForestClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
        'HoeffdingTreeClassifier': HoeffdingTreeClassifier,
    }

    if older_one_metadata['model_type'] == 'sklearn':
        constructor = cons[older_one_metadata['constructor']]
        model_params = older_one_metadata['model_hyperparameters']
        new_model, = train_SKLearn(constructor=constructor, model_params=model_params, **{'id': older_one_metadata['id']})()(data)

    # reassamble models
    trained_models, = addModel(good_ones, new_model)

    soil.alias('laura_pred', trained_models)


def RetrainingWeights():
    model = soil.data('laura_pred')
    print("Models's metadata:", model.metadata)

    data = soil.data('iris')
    print("Data's metadata:", data.metadata)
    
    print("Let's do it!")

    print("Let's see how many models we have and their ids!")
    model_ids, = getting_ids(model)
    
    # print(model_ids.data)
    # soil.alias('model_ids', model_ids)

    # print('My IDS: ', model_ids.metadata)

    ids = model_ids.data
    new_weights = []
    for _id in ids:
        print('this is the current id:', _id)
        # Taking only a small model
        small_model, = taking_small(model, _id=_id)  # taking_small -> filter_by_id
        print('Im a little model: ', small_model.metadata)

        # predictions for the single model
        y_pred, = make_predictions(small_model, data)
        print('predicted!!!')
        print('predictions: ', y_pred.metadata)

        # evaluation of the single model
        evaluation, = get_evaluation(y_pred, data, evaluate_function=accuracy_score)
        print('Evaluated, my accuray is: ', evaluation.data)

        soil.alias('stuuuff', evaluation)

        new_weights.append(evaluation.data[0]['accuracy_score'])

    model.metadata['weights'] = new_weights
    print(model.metadata)


def main():
    ''' Argument parsing. '''
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # evaluate_model(**vars(args))
    # adaptation_policy = 'KickOutworst'
    # adaptation_policy = 'KickOutolder'
    adaptation_policy = 'RetrainingWeights'
    if adaptation_policy == 'KickOutworst':
        kickoutWorst()
    elif adaptation_policy == 'KickOutolder':
        kickoutOlder()
    elif adaptation_policy == 'RetrainingWeights':
        RetrainingWeights()
    else:
        print('NOT IMPLEMENTED')
        raise NotImplementedError()


    # RetrainingWeights()


if __name__ == '__main__':
    main()
