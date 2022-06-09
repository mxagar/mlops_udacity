'''This module performs the necessary data science processing steps
to build and save models that predict customer churn using the
Credit Card Customers dataset from Kaggle:

https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code

These are the steps which are carried out:
- The dataset is loaded
- Exploratory Data Analysis (EDA)
- Feature Engineering (FE)
- Training: Random Forest and Logistic Regression models are fit
- Classification report plots

Clean code principles are guaranteed:
- Modularized code
- PEP8 conventions
- Error handling
- Testing is carried in the companion file: churn_script_logging_and_test.py

PEP8 conventions checked with:

>> pylint churn_library.py # 8.30/10
>> autopep8 churn_library.py

The file can be run stand-alone:

>> python customer_churn.py

The script expects the proper dataset to be located in `./data`

Additionally:

- The produced models are stored in `./models`
- The plots of the EDA and the classification results are stored in `./images`

Author: Mikel Sagardia
Date: 2022-06-08
'''

import os
#os.environ['QT_QPA_PLATFORM']='offscreen'
import joblib

#import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # It needs to be called here, otherwise we get error!
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

#from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

# Set library/module options
os.environ['QT_QPA_PLATFORM']='offscreen'
matplotlib.use('TkAgg')
sns.set()

def import_data(pth):
    '''Returns dataframe for the csv found at pth.

    input:
            pth (str): a path to the csv
    output:
            df (pandas.DataFrame): pandas dataframe with the dataset
    '''
    try:
        data = pd.read_csv(pth)
        return data
    except (FileNotFoundError, NameError):
        print("File not found!")
    except IsADirectoryError:
        print("File is a directory!")

    return None

def perform_eda(data):
    '''Performs EDA on df and saves figures to `images/` folder
    input:
            df (pandas.DataFrame): dataset

    output:
            None
    '''
    # General paramaters
    figsize = (20,15)
    dpi = 200
    rootpath = './images/eda'

    # New Churn variable: 1 Yes, 0 No
    data['Churn'] = data['Attrition_Flag'].apply(lambda val:
                                                 0 if val == "Existing Customer" else 1)
    # Figure 1: Churn distribution (ratio)
    fig = plt.figure(figsize=figsize)
    data['Churn'].hist()
    fig.savefig(rootpath+'/churn_dist.png', dpi=dpi)

    # Figure 2: Age distribution
    fig = plt.figure(figsize=figsize)
    data['Customer_Age'].hist()
    fig.savefig(rootpath+'/age_dist.png', dpi=dpi)

    # Figure 3: Marital status distribution
    fig = plt.figure(figsize=figsize)
    data['Marital_Status'].value_counts('normalize').plot(kind='bar')
    fig.savefig(rootpath+'/marital_status_dist.png', dpi=dpi)

    # Figure 4: Total transaction count distribution
    fig = plt.figure(figsize=figsize)
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve
    # obtained using a kernel density estimate
    sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True)
    fig.savefig(rootpath+'/total_trans_ct_dist.png', dpi=dpi)

    # Figure 5: Correlations
    fig = plt.figure(figsize=figsize)
    sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    fig.savefig(rootpath+'/corr_heatmap.png', dpi=dpi)


def encoder_helper(data, category_lst, response="Churn"):
    '''Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15
    input:
            df (pandas.DataFrame): dataset
            category_lst (list of str): list of columns that contain
                categorical features
            response (str): string of response name [optional argument
                that could be used for naming variables or index y column]

    output:
            df (pandas.DataFrame): pandas dataframe with new columns
            cat_columns_encoded (list of str): names of new columns
    '''

    # Names of new encoded columns
    cat_columns_encoded = []

    # Automatically detect categorical columns
    if not category_lst:
        category_lst = list(data.select_dtypes(['object']).columns)

    # Loop over all categorical columns
    # Create new variable which contains the churn ratio
    # associated with each category
    for col in category_lst:
        col_lst = []
        col_groups = data.groupby(col).mean()[response]

        for val in data[col]:
            col_lst.append(col_groups.loc[val])

        col_encoded_name = col + "_" + response
        cat_columns_encoded.append(col_encoded_name)
        data[col_encoded_name] = col_lst

    return data, cat_columns_encoded


def perform_feature_engineering(data, response="Churn"):
    '''
    input:
              df (pandas.DataFrame): dataset
              response (str): string of response name
                [optional argument that could be used
                for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # New Churn variable: 1 Yes, 0 No
    try:
        assert 'Attrition_Flag' in data.columns
        data[response] = data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
    except AssertionError as err:
        print("The df must contain the column 'Attrition_Flag'.")
        raise err
    except KeyError as err:
        print("Response key must be a string!")
        raise err

    # Traget variable
    y = data[response]

    # Drop unnecessary columns
    try:
        data.drop('Attrition_Flag', axis=1, inplace=True)
        data.drop('Unnamed: 0', axis=1, inplace=True)
        data.drop('CLIENTNUM', axis=1, inplace=True)
        #data.drop(response, axis=1, inplace=True)
    except KeyError as err:
        print("Missing columns in the dataframe.")
        raise err

    # Automatically detect categorical columns
    category_lst = list(data.select_dtypes(['object']).columns)

    # Encode categorcial variables as category ratios
    cat_columns_encoded = []
    data, cat_columns_encoded = encoder_helper(data, category_lst, response)

    # Drop target 
    data.drop(response, axis=1, inplace=True)

    # Automatically detect numerical columns    
    quant_columns = list(data.select_dtypes(['int64','float64']).columns)
    # Features: categorcial + numerical
    # but all identified as numerical now, because we encoded them so!
    # So, WRONG: keep_cols = quant_columns + cat_columns_encoded
    for col in cat_columns_encoded:
        try:
            assert col in quant_columns
        except AssertionError as err:
            print(f"Column {col} not found in set of numerical columns.")
            raise err
    keep_cols = quant_columns
    X = data[keep_cols]

    try:
        assert X.shape[1] == 19
    except AssertionError as err:
        print(f"Wrong number of columns: {X.shape[1]} != 19")
        raise err

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                outputpath):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train (pandas.DataFrame): training response values
            y_test (pandas.DataFrame):  test response values
            y_train_preds_lr (np.array): training preds. from log. regression
            y_train_preds_rf (np.array): training preds. from random forest
            y_test_preds_lr (np.array): test preds. from logistic regression
            y_test_preds_lr (np.array): test preds. from logistic regression
            y_test_preds_rf (np.array): test preds. from random forest

    output:
             None
    '''
    # General parameters
    dpi = 200
    figsize = (15, 15)

    # Unpack
    y_train_preds_lr, y_train_preds_rf = y_train_preds
    y_test_preds_lr, y_test_preds_rf = y_test_preds

    # Random forest model: Classification report
    fig = plt.figure(figsize=figsize)
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    fig.savefig(outputpath+'/rf_classification_report.png', dpi=dpi)

    # Logistic regression model: Classification report
    fig = plt.figure(figsize=figsize)
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    fig.savefig(outputpath+'/lr_classification_report.png', dpi=dpi)

def roc_curve_plot(models, X_test, y_test, outputpath):
    '''
    creates and stores the feature importances in pth
    input:
            models (objects): trained models
            X_test (numpy.array): test split features to compute ROC curves
            y_test (numpy.array): test split target to compute ROC curves

    output:
             None
    '''
    # General parameters
    dpi = 200
    figsize = (15, 8)

    # Unpack models
    lrc, rfc = models # lrc, cv_rfc.best_estimator_

    # ROC Plots
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    rfc_plot = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    fig.savefig(outputpath+'/roc_plots.png', dpi=dpi)

def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(20,10))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labelss
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save plot
    fig.savefig(output_pth+'/feature_importance.png', dpi=600)

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model_best.pkl')
    joblib.dump(cv_rfc, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    #try:
    #    # Check models can be loaded
    #    rfc_model = joblib.load('./models/rfc_model_best.pkl')
    #    lr_model = joblib.load('./models/logistic_model.pkl')
    #except FileNotFoundError:
    #    print("Models could not be saved!")

    # Define image output path
    image_output_path = "./images/results"

    # Save classification report plots
    classification_report_image(y_train,
                                y_test,
                                (y_train_preds_lr,y_train_preds_rf),
                                (y_test_preds_lr,y_test_preds_rf),
                                image_output_path)

    # Pack models for the ROC curve computation
    models = (lrc, cv_rfc.best_estimator_)

    # Save ROC curve plots
    roc_curve_plot(models,
                   X_test,
                   y_test,
                   image_output_path)

    # Save plot of feature importance
    feature_importance_plot(cv_rfc, X_train, image_output_path)

if __name__ == "__main__":

    # Load dataset
    print("Loading dataset...")
    FILEPATH = "./data/bank_data.csv"
    df = import_data(FILEPATH)

    # Perform Exploratory Data Analysis (EDA)
    print("Performing EDA...")
    perform_eda(df)

    # Perform Feature Engineering
    print("Performing Feature Engineering...")
    RESPONSE = "Churn"
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, response=RESPONSE)

    # Train the models
    # Models are saved to `models/`
    # and reports saved to `images/`
    print("Trainig...")
    train_models(X_train, X_test, y_train, y_test)
