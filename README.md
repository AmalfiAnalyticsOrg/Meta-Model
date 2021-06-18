# Adaptative Learning

The goal structure of the module will be based on four main scripts:
- Train
- Incremental Train
- Predict 
- Model Evaluation

## Train
This script recieves:

- A **soil dataset (DS)** containing the training data

- A *@trainable* **module** that will create the instance of the small models given the desired model, its parameters and an Id for each model. The *@trainable* module creates a list of the model and can both instantiate and/or train for the first time. Therefore, one could generate new small models by importing them in the train script as long as they have the same structure of a SKlearn model.

  This module is called train_SKLearn, it can instance any kind of ML class which has the same architecture and structure of a SKLearn model.

- Two extra *@modulify*  **modules** that train the whole metamodel for the first time and return the DS of it given **its parameters**, the **DS** of the instancied small models and the data. 

  The first module is the assembling which recieves the small models instances and the training parameters and also recives the training parameters and data because this is, actually, the function which trains the small models.

  The second module is the training one which recived the trained small models, data and training parameters and trains the MetaModel with its weights and specifications in a Soil DS.

```python
small_model, = train_SKLearn(constructor, model_params={...}, _id)
predictor_ref = assembling(models=models, **training_params)()(data)
predictor_ref, = training(data, predictor_ref, **training_params)
```

The implemented **small models** are: *Decision Tree regression, Decision Tree classifier, Random Forest, LinearDiscriminantAnalysis and Hoeffding Tree Classifier*.  



The **model_params** is a dictionary of the model hyperparameters where keys are the name of the hyperparameters and values their corresponding values. 



Take into account that all the modules return a Soil DS.



*@trainable* is a  higher-order function, i.e.,  a function that does at least one of the following:

- takes one or more functions as arguments,
- returns a function as its result.

which unifies the way the trained modules are called in the API. It allows us to specify the common things that we are interested in doing in every training such as: the train-test split or cross-validation.

In this case, the higher-order function is a decorator, a function that takes another function and extends the behavior of the latter without explicitly modifying it, of the trainining small model functions  and it is executed before them allowing the user to do partial executations. For instance, we can only give some parameters to *@trainable* instead of all the requiered ones and we will obtain the inicialization, the instance of the desired model but not the trained model itself.

The *@trainable* decorator has the follong structure:

```python
def trainable(module):
    def decorator_1(**model_params):
        def train_fn(**train_params):
            def inner_module(*data):
                return train(*data, module=module, model_params=model_params, **train_params)
            return inner_module
        return train_fn
    return decorator_1
```

The trainable function recives a soil module and returns the decorator itself.

The decorator_1 function recives a dictionary with the hyperparameters of the model and returns the instanced models.

The train_fn function recives a dictionary with the hyperparameters of the model and returns the instanced models.

And the inner_module recieves the data and returns the trained models.



## Incremental Train

This script recieves a **dataset** (DS), a **trained model** (DS), and some modules with its parameters: **adaptation function** and **its parameters**, **evaluation function** and returns a **DS trained model**.

```python
trained_model, = incremental_train(trained_model, new_data, adaptation_fun=KickOutWorst, module_params={...}
                                  evaluation_fun=Accuracy, evaluation_params={...})

@modulify(output_types=[Metamodel])
def incremental_train(model, data, adaptation_fun=None, module_params=None):
     incremented_model, = module(model, data, **module_params)
     return [incremented_model]

@modulify(output_types=[Metamodel])
def adaption_fun(data, models):
    for model in models:
        score, = evaluate(model)
    return [MetaModelDS(trained_models, metadata={...})]

incremental_train(data, my_meta_model, module=adaptation_fun, module_params={...})
```

The implemented **adaptation_fun** are: *Kick out worst* , *Kick out older* and *Retraining Weights*. All of them can be found in: `./modules/retraining`. These functions define the followed strategy to update the model. Therefore, one could use different adaptation functions by creating them. There are two main conditions to generate these functions. On the one hand, we must ensure that they receive data and the models both in a DS format. And, on the other hand, these functions must return a MetaModelDS in a DS format.

- *Kick out worst*: given the MetaModel and the small models which compose it, it takes the worst model and removes it. After that, it retrains the model that has been deleted with the new available data and adds it to the metamodel.
- *Kick out older*: given the MetaModel and the small models which compose it, it takes the older model and removes it. After that, it retrains the model that has been deleted with the new available data and adds it to the metamodel.
- *Retraining Weights*: given the MetaModel and the small models which compose it, it takes the weights assigned to each small model and retrains them by evaluating each small model again and assigning to each small model some weight according to its performance as in the training phase. 

The **module_params** is a dictionary of the adaptation_fun hyperparameters where keys are the name of the hyperparameters and values their corresponding values.

## Predict
This funciton recieves a **trained model** (as a DS), a **dataset** (as a DS) and a **predict function** and returns a **DS database**. 
```python
@modulify(output_types= lambda *inputs, **kwargs [inputs[1].__class__])
def predict(model, data, predict_fun=None, predict_params=None):
  	predictions, = get_predictions(model, data, **predict_params)
    return [MyDS(predictions, metadata)]
```

The **predict_fun** is the function used to predict. There are two main types of functions. If the model used is a simple model, for instance, *random forest, k-means*... it will just fit the model. However, if the model used to predict is the *MetaModel* the *pred_fun* will be the technique used to weigh the models included in it. The implemented *predic_fun* are: *major voting, best guy, weighted mean* and *weighted mode*. All of them can be found in: `./modules/prediction_functions/`. In addition, one could use different prediction functions by creating them. There are two main conditions to generate these functions. On the one hand, we must ensure that they receive data and the models both in a DS format. And, on the other hand, these functions must resturn a database in a DS format.

The **predict_params** is a dictionary of the prediction function hyperparameters where keys are the name of the hyperparameters and values their corresponding values. 

## Evaluate model

This script recieves a **trained model** (DS), a **dataset** (DS), a module that is the **predict function** with its parameters and an **evaluation function** with its parameters; and returns **evaluation metrics** with an identifier of time. 
```python
@modulify()
def evaluate_model(model, data, evaluation_fun=None, evaluation_params=None, predict_fun=None, predict_params=None):
    # unserialize fun
    predictions, = predict_fun(model, data, **predict_params)
    metrics, = evaluation_fun(predictions, **evaluation_params)
    return[MyDS(metrics, metadata)]
```

The **evaluation_fun** is the function used to evaluate the model. This function returns a set of metrics: Accuracy, Precision, Recall, *R Square*, *Mean Square Error (MSE), Mean Absolut Error (MAE)*, *F1 score* and *Binary Crossentropy*. This function is saved in  `./modules/evaluation_functions/`. If one wishes to use some other metric it can be implemented and modified in this function.

The **evaluation_params** is a dictionary of the evaluation function hyperparameters where keys are the name of the hyperparameters and values their corresponding values. 

The **predict_fun** is the function used to predict. There are two main types of functions. If the model used is a simple model, for instance, *random forest, k-means*... it will just fit the model. However, if the model used to predict is the *MetaModel* the *pred_fun* will be the technique used to weigh the models included in it. The implemented *predic_fun* are: *major voting, best guy, weighted mean* and *weighted mode*. All of them can be found in: `./modules/prediction_functions/`. In addition, one could use different prediction functions by creating them. There are two main conditions to generate these functions. On the one hand, we must ensure that they receive data and the models both in a DS format. And, on the other hand, these functions must resturn a database in a DS format.

The **predict_params** is a dictionary of the prediction function hyperparameters where keys are the name of the hyperparameters and values their corresponding values. 

## Structure

The structure of this architecture is organised as follows:

1. **Data Structures:** set of models (models, metamodel and adaptative models) saved as a soil data structure. 
2. **Modules:** set of folders where each module encapsulates a specific function that can  act on data in the workspace. They provide tools fot the evaluation, prediction and adpatation functions. 
3. **~new data** (ignore for now)
4. **~Scheduler:** Python scripts to call Data Structures and Modules. These scripts can be called by the user as he desires or they can be run automatically every X times.

```makefile
adaptative
│   README.md
│   __init__.py   
│
└───data_structures
│   │   __init__.py
│   │   es_data_structure.py
│   │   predictions.py
│   │   sklearn_data_structures.py
│   │   baseline_model.py
│   
└───lib
│   │   __init__.py
│   │   utils.py
│
└───modules
│   │   __init__.py
│   │   to_es_data_structure.py
│   │   trainable.py
│   │   make_predictions.py
│   │
│   └───evaluation_functions
│   │   │   evaluating_functions.py
│   │      
│   └───prediction_functions
│   │   │   prediction_functions.py
│   │   │   │    weighted_majority_bin
│   │   │   │    weighted_majority
│   │   
│   └───adaptation_functions
│   │   │   adapt.py
│   │   
│   └───static
│   │   │   baseline_class.py
│   │   │   MM_methods.py
│   │   │   train_Baseline.py
│   │   │   train_Historical.py
│   │   │   train_MM.py
│   │   │   train_SKLearn.py
│   │ 
│
└───new_data
│   │   __init__.py
│   │   new_data.py
│
└───scheduler
│   │   __init__.py
│   │   train.py
│   │   predict.py
│   │   evaluate_model.py
│   │   evaluate_predictions.py
│
└───setup
│   │   __init__.py
│   │   reset_dbs.py
```





## Metadata

The code assums data and metadata for each soil data structure, i.e. each DS has the .data and .metadata atributes. 

### Data 

The implemented models assume that the given data will have a metadata attribute (a dictionary) which includes, at least:

- \['columns']['y_labes'] : variable that we want to estimate or predict
- \['columns']['x_labes'] : set of variables aviables which can be potentially used to train the models.

### Models

The trained models contain the model itself in the data attribute and metadata in the model_metadata attirbute. 

*'Model_metadata'* is a dictionary which includes the following parameters:

- **x_labels** : variables used by the model

- **y_labels** : predicted variable by the model
- **time** : nanoseconds since the epoch (integer)
- **accuracy** : accuracy of the model
- **model_type** : 'sklearn', 'amalfi-metamodel' (string)


