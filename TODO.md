---
step: 
    1. fix the definition of what model to use based on the linear assumptions. ⚠️
        a. the definitions to regression and classification problems should be distinct. the testing for regression is the one in place now, but it doesn't work at all for classification since the assumptions are very different. ⚠️
        b. figure out in detail what are the assumptions for classification and why they are so. most problems are classification anyway. ⚠️
    2. reassign column types and make it so the class (auto_definitions/ModelDataProfiler) makes the type assignments by itself -- no passing it as input. ✅
    3. make the poorly inputed features get fixed (e.g. percentage having the % within the feature values) ✅
    4. return the recommendation of model type to be used so model_definitions can use it to train a set of candidates. ⚠️
    5. the model_definitions class should receive a dataframe for training. as a parameters {
        z. This version is one-hot encoding features that simply don't make any sense (e.g  dunning activities, activity_type_hystory -- to many unique values for one-hot-encoding. fix it) ✅
        a. the type of training validation to be used -- cross validation, etc.
        b. loss function ✅
        c. hyperparameter search function with hyperopt as standard ✅
    }
    as atributes {
        a. the model itself ✅
        b. comparison of status quo vs. model results for train and validation sets.
        c. parameters used for training ✅
        d. dataset used. ✅
        e. model type, feature names and feature types, feature importance. ✅
        f. metadata as json-- time of training, name and place of the source file used for training, number of observations for training, validation, date of training. ⚠️
    
    } ⚠️
    6. load the model, choosing raw or preprocessed data, and data scorer.
        a. select type of output, use csv for this early version, just as a means to uploading to kaggle. ⚠️

    7. IMPROVEMENTS
    {
    1. USE ([GradientFeatureSelector](https://nni.readthedocs.io/en/stable/feature_engineering/gradient_feature_selector.html)) for feature selection instead of whatever we are doing currently. seems very promissing ⚠️⚠️⚠️⚠️⚠️ Tried this for a while. The tensor transformations and type mismatchs -- present in this version -- kept me from moving forward. I am adding a patch version for the tensors assignments, but there's still the issue of loss being defined as NaN. I will work on this feature later. Let's focus now in the order in which things are listed here. Starting from the improvement instead of the current features was not a good call. Get this working, then make another version. Classic Me, though. ⚠️⚠️⚠️⚠️⚠️
    } ⚠️
---
To do list for this version.
labels: 
    ⚠️: being done
    ✅: done