# ML My Flow
Simple application to povide AutoML and MLOpls features.

It's a Docker application, Python based with Flask, Pandas and Sklearn.

## Getting Started
Prerequisites: [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [Docker](https://docs.docker.com/engine/install/).

### Steps:
1. Clone the repository.
   
    Open the terminal, go to the folder that you want to save the files and type: 

    ```git clone https://github.com/evandropp10/ml_my_flow.git```

2. Build the container.
   
    Using the terminal go to the repository folder, should have the file Dockerfile, and type:

    ```docker image build -t ml-my-flow .```

3. Run the container.
   
    Using the terminal type:

    ```docker run -p 5000:5000 -d ml-my-flow```

4. Check the container Status.
   
    Wait some seconds and check if the container status is UP with this command:

    ```docker ps```

The Application is running on port 5000.

## Endpoints

### Check API Status
**get /**

Return 200 if API is available.

### Dataset Prepare
**post /prepare**

In this endpoint are executed 4 basic data preparation.
1. Convert all the object columns to integer.
2. Remove columns with high variation. 
3. Remove columns with null.
4. Decompose Datetime columns in 3 columns: year, month and day.
   
#### Input parameters:
  
```post_data = {'max_na': 0.2, 'max_var': 0.2, 'payload': df_json}```
* **max_na:** *Float*. Maximum acceptable percentage of null in a column. If more the column is removed.
* **max_var:** *Float*. Maximum acceptable percentage of value variation in a column. It's based on the concept that a high number of unique values couldn't be good for prediction.
* **payload:** *Json*. The dataset in format json.

#### Output:
* Dataset in format json.


### Train and Test
**post /train_test**

In this endpoint the dataset are submitted to 3 different algorithms. It's possible to use regression or classification methods.

#### Input parameters:
  
```post_data = {'train_columns': ','.join(train_columns), 'target_column': target, 'ml_method': ml_method, 'test_size': 0.2, 'payload': df_json}```
* **train_columns:** *String*. Columns separated by comma to train the models, if "" all the columns will be used. Example: "col1, col2, col3".
* **target_column:**  *String*. Column to be predicted.
* **ml_method:** *String*. Column to choose the machine learning method. Should be "Regression" or "Classification".
* **test_size:** *Float*. Size of the part of the data to be tested with the trained algorithms.
* **payload:** *Json*. Dataset.

#### Output:
* Json with the accuracy result of the 3 different algorithms.

### Train
**post /train**

This endpoint is for training the model with the choosen algorithm. The trained model is saved in pickle format on the folder *model_registry* inside the container.

#### Input parameters:
  
```post_data = {'train_columns': ','.join(train_columns), 'target_column': target, 'ml_method': ml_method, 'model_name': 'gradient', 'payload': df_json}```
* **train_columns:** *String*. Columns separated by comma to train the models, if '' all the columns will be used. Example: "col1, col2, col3".
* **target_column:**  *String*. Column to be predicted.
* **ml_method:** *String*. Column to choose the machine learning method. Should be "Regression" or "Classification".
* **model_name:** *String*. Name of the algorithm to train the model. In Regression should be "linear", "gradient" or "randon_forest" and in Classification should be "KNN", "gradient" or "randon_forest".
* **payload:** *Json*. Dataset.

#### Output:
* Json with the pickle path.

### Predict
**post /predict**

This endpoint is for submit new data to predict the traget value.

#### Input parameters:
  
```post_data = {'model_path': model_path, 'payload': df_json}```
* **model_path:** *String*. Path of the pickle file inside the container.
* **payload:** *Json*. Dataset.

#### Output:
* Json with the prediction.

## Examples
Please check the folder /examples/. There are 2 example notebooks, one for regression and another for classification.

## AWS Architecture
Look at the folder /aws_architecture/ to find the architeture draw and explanation.
  








