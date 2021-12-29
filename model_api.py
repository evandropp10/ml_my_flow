from flask import Flask, request, jsonify
import pandas as pd
import pickle
import ml_my_flow as ml


# Function that create the app 
def create_app(test_config=None ):
    # create and configure the app
    app = Flask(__name__)

    @app.route('/prepare', methods=['POST'])
    def prepare(): 
        
        max_na = float(request.form.get('max_na'))
        max_var = float(request.form.get('max_var'))
        df_json = request.form.get('payload')
        
        df = pd.read_json(df_json)

        df_prep = ml.df_prepare(df, max_na=max_na, max_var=max_var)

        json_prep = df_prep.to_json(orient='records')

        return json_prep

    @app.route('/train_test', methods=['POST'])
    def train_test(): 
        train_columns = request.form.get('train_columns')
        target_column = request.form.get('target_column')
        ml_method = request.form.get('ml_method')
        test_size = float(request.form.get('test_size'))
        df_json = request.form.get('payload')
        
        if len(train_columns) > 0:
            train_columns = train_columns.split(',')

        df = pd.read_json(df_json)

        return_train = ml.train_test_models(df, train_columns=train_columns, target_column=target_column, ml_method=ml_method, test_size=test_size)

        return return_train

    @app.route('/train_model', methods=['POST'])
    def train_model(): 
        train_columns = request.form.get('train_columns')
        target_column = request.form.get('target_column')
        ml_methood = request.form.get('ml_method')
        model_name = request.form.get('model_name')
        df_json = request.form.get('payload')

        if len(train_columns) > 0:
            train_columns = train_columns.split(',')

        df = pd.read_json(df_json)

        return_train =  ml.train_model(df, train_columns=train_columns, target_column=target_column, ml_method=ml_methood, model_name=model_name)

        return return_train


    @app.route('/predict', methods=['POST'])
    def predict(): 
        model_path = request.form.get('model_path')
        df_json = request.form.get('payload')

        df = pd.read_json(df_json)
        
        result = ml.predict(df, model_file=model_path)

        json_result = pd.DataFrame(result, columns=['prediction']).to_json()
        
        return json_result

    @app.route('/')
    def available_test(): 
        return jsonify({
           "status": "success",
            "message": "API available"
        }) 
     
    return app

APP = create_app()

if __name__ == '__main__':
    #APP.run(host='localhost', port=5000, debug=True)
    APP.run(host='0.0.0.0')