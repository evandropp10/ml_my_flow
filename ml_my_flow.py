import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
from datetime import datetime
import os
import json
      

def df_prepare(df_train, max_na=0.2, max_var=0.2):
    
    train_columns = list(df_train.columns)
    
    ## Removing NaN columns if higher than max_na parameter
    cols_remove = []
    for col in train_columns:
        if df_train[col].isnull().sum() / len(df_train) > max_na:
            cols_remove.append(col)
            
    df_train = df_train.drop(columns=cols_remove, axis=1)

    train_columns = list(df_train.columns)

    # Puting number zero in NaN
    df_train = df_train.fillna(0)

    ## Categorize string columns
    for col in train_columns:
        if str(df_train[col].dtype) == 'object':
            df_train[col] = df_train[col].astype('category').cat.codes

    ## Removing high variability columns
    cols_remove = []
    for col in train_columns:
        if 'int' in str(df_train[col].dtype):
            if df_train[col].nunique() / len(df_train) > max_var:
                cols_remove.append(col)
            
    df_train = df_train.drop(columns=cols_remove, axis=1)

    train_columns = list(df_train.columns)

    ## Prepare Datetime columns
    for col in train_columns:
        if 'date' in str(df_train[col].dtype):
            df_train['year_' + col] = df_train[col].apply(lambda x: x.year)
            df_train['month_' + col] = df_train[col].apply(lambda x: x.month)
            df_train['day_' + col] = df_train[col].apply(lambda x: x.day)

            df_train = df_train.drop(columns=[col], axis=1)

    return df_train

def train_test_models(df_train_prep, train_columns='', target_column='', ml_method='', test_size=0.2):
    ## Basic validations
    try:
        validation = validation_parameters(df_train_prep, train_columns, target_column, ml_method)

        if validation != 'ok':          
            raise ValueError(validation)
    except ValueError as e:
        return e

    if train_columns == '':
        x = list(df_train_prep.columns)
        x.remove(target_column)
    else:
        x = train_columns

    y = target_column

    x_train, x_test, y_train, y_test = train_test_split(df_train_prep[x], df_train_prep[y], test_size=test_size, random_state=101)

    if ml_method == 'Regression':
        # Linear
        print('Linear')
        linear_result = regressor(x_train, x_test, y_train, y_test, operation='train_test', alg='linear')

        # Gradient Boost
        print('Gradient')
        gradient_result = regressor(x_train, x_test, y_train, y_test, operation='train_test', alg='gradient')
    
        # Randon Forest
        print('Randon Forest')
        randon_forest_result = regressor(x_train, x_test, y_train, y_test, operation='train_test', alg='randon_forest')

        result = {
                    "linear": linear_result,
                    "gradient": gradient_result,
                    "randon_forest": randon_forest_result
                }
        
    if ml_method == 'Classification':
        # KNN
        print('KNN')
        linear_result = classifier(x_train, x_test, y_train, y_test, operation='train_test', alg='KNN')

        # Gradient Boost
        print('Gradient')
        gradient_result = classifier(x_train, x_test, y_train, y_test, operation='train_test', alg='gradient')
    
        # Randon Forest
        print('Randon Forest')
        randon_forest_result = classifier(x_train, x_test, y_train, y_test, operation='train_test', alg='randon_forest')

        result = {
                    "KNN": linear_result,
                    "gradient": gradient_result,
                    "randon_forest": randon_forest_result
                }

    return result

def train_model(df_train_prep, train_columns='', target_column='', ml_method='', model_name=''):
    ## Basic validations
    try:
        validation = validation_parameters(df_train_prep, train_columns, target_column, ml_method)

        if validation != 'ok':          
            raise ValueError(validation)
    except ValueError as e:
        return e

    if train_columns == '':
        x = list(df_train_prep.columns)
        x.remove(target_column)
    else:
        x = train_columns

    y = target_column

    x_train = df_train_prep[x]
    y_train = df_train_prep[y]
    x_test = pd.DataFrame()
    y_test = pd.DataFrame()
    

    if ml_method == 'Regression':

        result = regressor(x_train, x_test, y_train, y_test, operation='train', alg=model_name)
        
    if ml_method == 'Classification':
        result = classifier(x_train, x_test, y_train, y_test, operation='train', alg=model_name)

    return result

def regressor(x_train, x_test, y_train, y_test, operation, alg):

    if alg == 'linear':
        model = LinearRegression()
    if alg == 'gradient':
        model = GradientBoostingRegressor()
    if alg == 'randon_forest':
        model = RandomForestRegressor()

    model.fit(x_train, y_train)
    
    if operation == 'train_test':
        prediction = model.predict(x_test)

        result = regressor_result(y_test, prediction)
                
    else:
        train_columns = list(x_train.columns)
        file_name, file_name_param = pickle_create(model, alg + '_regressor', train_columns)
        
        result = {"file_name": file_name,
                "file_name_param": file_name_param}

    return result

def classifier(x_train, x_test, y_train, y_test, operation, alg):

    if alg == 'KNN':
        model = KNeighborsClassifier()
    if alg == 'gradient':
        model = GradientBoostingClassifier()
    if alg == 'randon_forest':
        model = RandomForestClassifier()

    model.fit(x_train, y_train)
    
    if operation == 'train_test':
        prediction = model.predict(x_test)

        result = classifier_result(y_test, prediction)
                
    else:
        train_columns = list(x_train.columns)
        file_name, file_name_param = pickle_create(model, alg + '_classifier', train_columns)
        
        result = {"file_name": file_name,
                "file_name_param": file_name_param}

    return result

def predict(df, model_file):
    model = pickle.load(open(model_file, 'rb'))

    result = model.predict(df)

    return result

def pickle_create(model, model_name, train_columns):
    folder_name = 'models_registry/' + datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = folder_name + '/' + model_name + '.pickle'
    
    if os.path.exists(folder_name) == False:
        os.makedirs(folder_name)

    pickle.dump(model, open(file_name, 'wb'))

    params = {'columns': train_columns}
    file_name_param = folder_name + '/' + model_name + '.json'
    with open(file_name_param, 'w') as json_file:
        json.dump(params, json_file)

    return file_name, file_name_param

def regressor_result(y_test, prediction):
    df_result = pd.DataFrame(columns=['Test', 'Prediction'])
    df_result['Test'] = y_test
    df_result['Prediction'] = prediction

    result = {
                "MAE": metrics.mean_absolute_error(df_result['Test'],df_result['Prediction']),
                "MSE": metrics.mean_squared_error(df_result['Test'],df_result['Prediction']),
                "RMSE": metrics.mean_squared_error(df_result['Test'],df_result['Prediction'], squared=False)
            }
        
    return result

def classifier_result(y_test, prediction):
    df_result = pd.DataFrame(columns=['Test', 'Prediction'])
    df_result['Test'] = y_test
    df_result['Prediction'] = prediction

    result = {
                "Accuracy Score": metrics.accuracy_score(df_result['Test'],df_result['Prediction'])

            }
        
    return result

def validation_parameters(df_train_prep, train_columns, target_column, ml_method):
    if ml_method != 'Regression' and ml_method != 'Classification' and ml_method != 'Clustering':
        return "ml_method should be 'Regression', 'Classification' or 'Clustering'"

    if len(train_columns) > 0:
        for col in train_columns:
            if col not in df_train_prep.columns:
                    return "Train column '" +col+ "' does not exists"

    if target_column not in df_train_prep.columns:
        return 'Target column does not exists'

    return 'ok'