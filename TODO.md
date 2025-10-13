---
step: 
    1. fix the definition of what model to use based on the linear assumptions. ⚠️
    2. reassign column types and make it so the class (auto_definitions/ModelDataProfiler) makes the type assignments by itself -- no passing it as input. ⚠️
    3. make the poorly inputed features get fixed (e.g. percentage having the % within the feature values) ⚠️
    4. return the recommendation of model type to be used so model_definitions can use it to train a set of candidates. ⚠️
    5. the model_definitions class should receive a dataframe for training. as a parameters {
        a. the type of training validation to be used -- cross validation, etc.
        b. loss function
        c. hyperparameter search function with hyperopt as standard
    }
    as atributes {
        a. the model itself
        b. comparison of status quo vs. model results for train and validation sets.
        c. parameters used for training
        d. dataset used.
        e. model type, feature names and feature types, feature importance.
        f. metadata as json-- time of training, name and place of the source file used for training, number of observations for training, validation, date of training.
    
    } ⚠️
    6. load the model, choosing raw or preprocessed data, and data scorer.
        a. select type of output, use csv for this early version, just as a means to uploading to kaggle. ⚠️

    7. IMPROVEMENTS
    {
    1. USE ([GradientFeatureSelector](https://nni.readthedocs.io/en/stable/feature_engineering/gradient_feature_selector.html)) for feature selection instead of whatever we are doing currently. seems very promissing
    } ⚠️
---
To do list for this version.
labels: 
    ⚠️: being done
    ✅: done