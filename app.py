# from flask import request
from flask import Flask, render_template, request, jsonify, send_from_directory

# from json_flask import JsonFlask
# from json_response import JsonResponse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys

# app = JsonFlask(__name__)
app = Flask(__name__)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# 修改保存模型和标准化器的路径
best_model_path = resource_path("best_mlp_model.pkl")
scaler_path = resource_path("scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    # data = json.loads(request.data)
    train_file_path = request.json.get('trainFilePath')
    if not train_file_path or not os.path.exists(train_file_path):
        return jsonify({'error': 'Invalid train file path'}), 400
        # return JsonResponse.fail(msg='error: Invalid train file path')
    
    data = pd.read_excel(train_file_path)
    X = data.drop(['DATE', 'WEATHER_TYPE'], axis=1)
    y = data['WEATHER_TYPE']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    param_grid = {
        'hidden_layer_sizes': [(50, 50, 50), (100, 50, 25), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 2000],
        'early_stopping': [True],
    }
    
    mlp = MLPClassifier(random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_

    # 保存模型和标准化器
    joblib.dump(best_model, best_model_path)
    joblib.dump(scaler, scaler_path)

    return jsonify({'message': 'Model trained and saved successfully', 'modelPath': best_model_path, 'scalerPath': scaler_path})
    # return JsonResponse.succuess(msg='Model trained and saved successfully', data={'modelPath': model_path, 'scalerPath': scaler_path})

@app.route('/test', methods=['POST'])
def test_model():
    # data = json.loads(request.data)
    test_file_path = request.json.get('testFilePath')

    if not test_file_path or not os.path.exists(test_file_path) or not best_model_path or not scaler_path:
        return jsonify({'error': 'Invalid parameters'}), 400
        # return JsonResponse.fail(msg='error: Invalid parameters')
    
    test_data = pd.read_excel(test_file_path)
    X_test = test_data.drop(['DATE', 'WEATHER_TYPE'], axis=1)
    y_test = test_data['WEATHER_TYPE']
    
    scaler = joblib.load(scaler_path)
    X_test = scaler.transform(X_test)
    
    best_model = joblib.load(best_model_path)
    predictions = best_model.predict(X_test)
    
    report = classification_report(y_test, predictions, output_dict=True)
    matrix = confusion_matrix(y_test, predictions).tolist()

    return jsonify({'classificationReport': report, 'confusionMatrix': matrix})
    # return JsonResponse.succuess(msg='Test successfully', data={'classificationReport': report, 'confusionMatrix': matrix})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=666, debug=True)