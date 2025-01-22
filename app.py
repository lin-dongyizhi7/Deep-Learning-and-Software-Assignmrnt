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
    train_file_path = request.json.get('trainFilePath')
    train_data = request.json.get('trainData')
    data = pd.DataFrame(train_data)
    # 转换数据类型
    numeric_columns = ['TEMP', 'TEMP_ATTRIBUTES', 'DEWP', 'DEWP_ATTRIBUTES', 'SLP', 'SLP_ATTRIBUTES',
                       'STP', 'STP_ATTRIBUTES', 'VISIB', 'VISIB_ATTRIBUTES', 'WDSP', 'WDSP_ATTRIBUTES',
                       'MXSPD', 'GUST', 'MAX_ATTRIBUTES', 'MIN_ATTRIBUTES', 'PRCP', 'FRSHTT', 'WEATHER_TYPE', 'SEASON']

    for col in numeric_columns:
        # 检查是否所有值都可以转换为float，否则保留原样或者转换为其他适当类型
        try:
            data[col] = data[col].astype(float)
        except ValueError:
            print(f"Column {col} contains non-numeric values and could not be converted to float.")

    data = data.dropna()

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

@app.route('/test', methods=['POST'])
def test_model():
    test_file_path = request.json.get('testFilePath')
    test_data = request.json.get('testData')
    data = pd.DataFrame(test_data)
    # 转换数据类型
    numeric_columns = ['TEMP', 'TEMP_ATTRIBUTES', 'DEWP', 'DEWP_ATTRIBUTES', 'SLP', 'SLP_ATTRIBUTES',
                       'STP', 'STP_ATTRIBUTES', 'VISIB', 'VISIB_ATTRIBUTES', 'WDSP', 'WDSP_ATTRIBUTES',
                       'MXSPD', 'GUST', 'MAX_ATTRIBUTES', 'MIN_ATTRIBUTES', 'PRCP', 'FRSHTT', 'WEATHER_TYPE', 'SEASON']

    for col in numeric_columns:
        # 检查是否所有值都可以转换为float，否则保留原样或者转换为其他适当类型
        try:
            data[col] = data[col].astype(float)
        except ValueError:
            print(f"Column {col} contains non-numeric values and could not be converted to float.")

    data = data.dropna()

    X_test = data.drop(['DATE', 'WEATHER_TYPE'], axis=1)
    y_test = data['WEATHER_TYPE']
    
    scaler = joblib.load(scaler_path)
    X_test = scaler.transform(X_test)
    
    best_model = joblib.load(best_model_path)
    predictions = best_model.predict(X_test)
    predictions = predictions.tolist()
    
    report = classification_report(y_test, predictions, output_dict=True)
    matrix = confusion_matrix(y_test, predictions).tolist()

    return jsonify({'classificationReport': report, 'confusionMatrix': matrix, 'predictions': predictions })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=666, debug=True)