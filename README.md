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
   
    Wait 2 or 3 minutes and check if the container status is UP with this command:

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
   
* Input parameters:
  
    ```post_data = {'max_na': 0.2, 'max_var': 0.2, 'payload': df_json}```
    * max_na: Maximum acceptable percentage of null in a column. If more the column is removed.
    * max_var: Maximum acceptable percentage of value variation in a column. It's based on the concept that a high number of unique values couldn't be good for prediction.
    * payload: The dataset in format json.

* Output:
  * Dataset in format json.


### Train and Test
**post /train_test**






