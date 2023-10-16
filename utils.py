import random
from time import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tqdm.notebook import tqdm
from sklearn.model_selection import (
    cross_val_score, 
    cross_validate, 
    train_test_split
)

from sklearn import svm, datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    auc,
    roc_curve, 
    classification_report,
    RocCurveDisplay, 
    confusion_matrix,
    ConfusionMatrixDisplay
)
import xgboost as xgb
import catboost


def get_metrics(y_true, y_pred, file_name=None, start_string=''):
    '''
    This function gets true and predicted labels and calculates metrics.
    Writes returned string to file if file_name parameter is not None
    
    Parameters:
    -----------
    y_true : numpy ndarray
        List of true labels 
    y_pred : numpy ndarray
        List of predicted labels
    file_name : str, default=None
        String with name of file to write metrics
    start_string : str, default=''
        String with name of file to write metrics. If None, metrics won't be written
        
    Returns:
    --------
    result_string : str
        String with full metrics information: F1, presicion, recall, accuracy, confusion matrix
    
    '''
    
    result_string = start_string + '\n'
    result_string += classification_report(y_true.astype(int), y_pred.astype(int))
#     print(type(y_p))
    cm = confusion_matrix(y_true, y_pred, labels=sorted(list(set(y_true))))
    result_string += "\n\n" + 'Confusion matrix:\n' + str(cm)
    
    if file_name is not None:
        with open(file_name, 'a') as file:
            file.write(result_string)
            file.write('\n\n\n')
    return result_string


def ml_class_train_template(
    X,
    y,
    model_class,
    args: dict,
    k_folds:int = 5,
    test_size:float = 0.33,
    feature_importances: bool = False,
    method: str = "choice",
    random_state=None
) -> tuple:
    models = {
        f"{i+1}_fold" : None for i in range(k_folds)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if method == "choice":
        y_pred = np.empty((k_folds, y_test.shape[0]))

    elif method == "proba":
        y_pred = np.empty((k_folds, y_test.shape[0], np.unique(y).shape[0]))

    cv = StratifiedKFold(n_splits=k_folds)
    if feature_importances:
        feature_importances_array = np.empty((k_folds, X_train.shape[-1]))

    for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        model = model_class(random_state=random_state, **args)
        model.fit(X_train[train], y_train[train])
        models[f'{i+1}_fold'] = model
        if method == "choice":
            y_pred[i] = model.predict(X_test).reshape(-1)

        elif method == "proba":
            y_pred[i] = model.predict_proba(X_test)

        if feature_importances:
            feature_importances_array[i] = model.feature_importances_

    # prediction of final results
    if method == "choice":
        y_pred_result = np.empty_like(y_test)
        for index in range(y_pred_result.shape[0]):
            unique_values, counts = np.unique(y_pred[:, index], return_counts=True)
            y_pred_result[index] = unique_values[counts.argmax()]

    elif method == "proba":
        y_pred = y_pred.sum(axis=0)
        y_pred_result = np.argmax(y_pred, axis=1)
        y_pred_result[y_pred_result == 0] = -1

    accuracy = accuracy_score(y_test, y_pred_result)
    f1 = f1_score(y_test, y_pred_result, average="weighted")

    metrics = {
        # "precision": precision,
        # "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1
    }

    if feature_importances:
        metrics["feature_importances"] = feature_importances_array.mean(axis=0)

    return metrics, models


def random_forest_train(
    data, 
    y, 
    k_folds=5,
    num_estimators=1000, 
    description='raman spectra dataset', 
    class_type='group', 
    calc_metrics=True,
    file_name=None,
    printing_metrics=True
):
    '''
    This function gets data to train and test and other parameters related with training, 
    cross-validates RandomForest and calculates metrics
    
    Parameters:
    -----------
    data : numpy ndarray
        Array with spectra features. 
        data.shape = (num_samples, num_features)
    y : numpy ndarray
        List of labels: number of group and class. 
        y.shape = (num_features, 2)
    k_folds : int, default=5
        Number of folds in cross-validation split
    num_estimators : int
        Number of estimators(trees, for example) or iterations during training
    description : str, deffault='raman spectra dataset'
        A dataset description or anything else, related with prediction task.
    class_type : str, default='group'
        Type of label: can be only 'group' or 'class'.
    calc_metrics : bool, default=True
        Parameter indicating whether metrics need to be calculated or not.
    file_name :  str, default=None
        String with name of file to write metrics. If None, metrics won't be written.
        
    Returns:
    --------
    metric_dict : dict
        Dictionary with calculated mean values of metrics: accuracy, f1 score, recall, precision.
    models : dict
        Dictionary with models per every fold, trained during cross-validation. 
    
    '''
    
    if class_type not in ('group', 'class'):
        raise Exception("class_type may be equal only 'group' or 'class'")
        
    num_features = data.shape[1]
    exrt_crystal_system = ExtraTreesClassifier(
        n_estimators=num_estimators, 
#         max_depth=40, 
        max_features=num_features, 
        n_jobs=-1, 
    #     random_state=random_state,
        warm_start=False
    )
    metric_dict, models = train_template(
        exrt_crystal_system, 
        data, 
        y, 
        k_folds=5, 
        description=description, 
        class_type=class_type, 
        calc_metrics=calc_metrics, 
        file_name=file_name,
        printing_metrics=printing_metrics,
    )
    return metric_dict, models
    
    
def xgboost_train(
    data, 
    y, 
    k_folds=5,
    num_estimators=1000, 
    description='raman spectra dataset', 
    class_type='group', 
    calc_metrics=True,
    file_name=None,
    printing_metrics=True
):
    
    '''
    This function gets data to train and test and other parameters related with training, 
    cross-validates XGBClassifier and calculates metrics
    
    Parameters:
    -----------
    data : numpy ndarray
        Array with spectra features. 
        data.shape = (num_samples, num_features)
    y : numpy ndarray
        List of labels: number of group and class. 
        y.shape = (num_features, 2)
    k_folds : int, default=5
        Number of folds in cross-validation split
    num_estimators : int
        Number of estimators(trees, for example) or iterations during training
    description : str, deffault='raman spectra dataset'
        A dataset description or anything else, related with prediction task.
    class_type : str, default='group'
        Type of label: can be only 'group' or 'class'.
    calc_metrics : bool, default=True
        Parameter indicating whether metrics need to be calculated or not.
    file_name :  str, default=None
        String with name of file to write metrics. If None, metrics won't be written.
        
    Returns:
    --------
    metric_dict : dict
        Dictionary with calculated mean values of metrics: accuracy, f1 score, recall, precision.
    models : dict
        Dictionary with models per every fold, trained during cross-validation. 
    
    '''
    
    if class_type not in ('group', 'class'):
        raise Exception("class_type may be equal only 'group' or 'class'")
    
    xg_clsfr = xgb.XGBClassifier(
#         objective='multi:softmax', 
#         learning_rate = 0.01,
        n_estimators=num_estimators, 
        eval_metric='merror'
    )
    metric_dict, models = train_template(
        xg_clsfr, 
        data, 
        y, 
        k_folds=5, 
        description=description, 
        class_type=class_type, 
        calc_metrics=calc_metrics, 
        file_name=file_name,
        printing_metrics=printing_metrics
    )
    return metric_dict, models
    
def catboost_train(
    data, 
    y, 
    k_folds=5,
    num_estimators=1000, 
    description='raman spectra dataset', 
    class_type='group', 
    calc_metrics=True,
    file_name=None,
    printing_metrics=True
):
    
    '''
    This function gets data to train and test and other parameters related with training, 
    cross-validates CatBoostClassifier and calculates metrics
    
    Parameters:
    -----------
    data : numpy ndarray
        Array with spectra features. 
        data.shape = (num_samples, num_features)
    y : numpy ndarray
        List of labels: number of group and class. 
        y.shape = (num_features, 2)
    k_folds : int, default=5
        Number of folds in cross-validation split
    num_estimators : int
        Number of estimators(trees, for example) or iterations during training
    description : str, deffault='raman spectra dataset'
        A dataset description or anything else, related with prediction task.
    class_type : str, default='group'
        Type of label: can be only 'group' or 'class'.
    calc_metrics : bool, default=True
        Parameter indicating whether metrics need to be calculated or not.
    file_name :  str, default=None
        String with name of file to write metrics. If None, metrics won't be written.
        
    Returns:
    --------
    metric_dict : dict
        Dictionary with calculated mean values of metrics: accuracy, f1 score, recall, precision.
    models : dict
        Dictionary with models per every fold, trained during cross-validation. 
    
    '''
    
    if class_type not in ('group', 'class'):
        raise Exception("class_type may be equal only 'group' or 'class'")
    
#     yes, I love cats
    kitty_model = catboost.CatBoostClassifier(
        iterations=num_estimators,
#         if you want to see logs, set 'veerbose' to 1 or 2
        verbose=0
    )
    
    metric_dict, models = train_template(
        kitty_model, 
        data, 
        y, 
        k_folds=5, 
        description=description, 
        class_type=class_type, 
        calc_metrics=calc_metrics, 
        file_name=file_name,
        printing_metrics=printing_metrics
    )
    return metric_dict, models
    
    
# is not neccesery:
def get_class_indexes(y, y_proba, needed_class, num_classes):
#     relables classes: 1 for needed class, 0 for other ones
    y_func = y.copy()
    y_proba_func = y_proba.copy()
    classes = np.array([i for i in range(num_classes) if i != needed_class])
#     print(classes)
    for i in classes:
        y_func[y_func == i] = -1
    y_func[y_func == needed_class] = 1
    y_func[y_func != 1] = 0 #for beautiness
    y_proba_func = 1 - (y_proba_func.sum(axis=1) - y_proba_func[:, needed_class])
    
    return y_func, y_proba_func


def print_roc(data, y,model, dataset_name, file_name, group_or_class='class', save_file=False):
    # Data shuffle
    new_data = data.copy()
    new_y = y.copy()
    seed = int(time())
    np.random.seed(seed=seed)
    np.random.shuffle(new_data)
    np.random.seed(seed=seed)
    np.random.shuffle(new_y)
    # Train data initialization
    X = new_data
    if group_or_class == 'class':
        new_y = new_y[:, 1]
    elif group_or_class == 'group':
        new_y = new_y[:, 0]
    else:
        raise Exception(f"parameter group_or_class should be equal 'class' of 'group',  not {group_or_class}")
    n_samples, n_features = X.shape
    n_classes = len(set(new_y))
    
    fig, ax = plt.subplots(figsize = (17, 9), dpi = 300)
    # fig.figsize = (17, 9)
    # fig.dpi = 300
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)

    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(n_splits=5)
    tpr = []
    roc_auc = []
    mean_auc = []
    
    for i, (train, test) in enumerate(cv.split(X, new_y)):
        model.fit(X[train], new_y[train])
        
        tpr_tmp = []
        roc_auc_tmp = []
        for index in range(n_classes):
            #     making scores and y_test without needed classes  
#             print(classification_report(new_y[test], model.predict(X)[test]))
            y_test, y_score = get_class_indexes(new_y[test], model.predict_proba(X)[test], index, 14)
            fpr_i, tpr_i, _ = roc_curve(y_test, y_score)
            
            tpr_tmp.append(np.interp(mean_fpr, fpr_i, tpr_i))
            roc_auc_tmp.append(auc(fpr_i, tpr_i))
        
        tpr.append(tpr_tmp)
        roc_auc.append(roc_auc_tmp)
        
        
    tpr = np.array(tpr)
    roc_auc = np.array(roc_auc)
    mean_tpr = tpr.mean(axis=0)
    mean_tpr[:, -1] = 1.0
    for tpr_i in mean_tpr:
        mean_auc.append(auc(mean_fpr, tpr_i))
        ax.plot(
            mean_fpr,
            tpr_i,
    #         color="b",
    #             label=f"ROC for class {index+1} (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.2,
        )
    std_auc = np.std(roc_auc)
    std_tpr = np.std(mean_tpr, axis=0)
    
    tprs_upper = np.minimum(mean_tpr.mean(axis=0) + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr.mean(axis=0) - std_tpr, 0)

    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.plot(
            mean_fpr,
            mean_tpr.mean(axis=0),
            color="r",
            label=f"Mean ROC (AUC = %0.4f $\pm$ %0.4f)" % (auc(mean_fpr, mean_tpr.mean(axis=0)), std_auc),
            lw=2,
            alpha=1,
        )

    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.01],
    )
    plt.ylabel("False positive rate", fontsize=15)
    plt.xlabel("True positive rate", fontsize=15)
    plt.title(f"ROC curve for {dataset_name} analysis", fontsize=20)
#     ylabel='False positive rate',
#     xlabel='True positive rate',
#     title="Receiver operating characteristic example"
    ax.legend(loc="lower right", fontsize=15)
    plt.grid()
    if save_file == True:
        plt.savefig(f"{file_name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
