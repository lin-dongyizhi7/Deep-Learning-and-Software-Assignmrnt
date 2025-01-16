import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys
import io
import contextlib

# 创建主窗口
root = tk.Tk()
root.title("Weather Classification Model Trainer and Tester")
root.geometry("900x450")

# 设置全局样式
style = ttk.Style()
style.theme_use('clam')  # 使用 'clam' 主题，提供更现代的外观
style.configure('TButton', font=('Arial', 12), padding=10)
style.configure('TLabel', font=('Arial', 12))
style.configure('TText', font=('Courier New', 10))

# 全局变量用于存储文件路径
train_file_path = None
test_file_path = None

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


# 选择训练文件
def select_train_file():
    global train_file_path
    train_file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if train_file_path:
        train_file_label.config(text=f"Train File: {train_file_path}")
    else:
        train_file_label.config(text="No train file selected")


# 选择测试文件
def select_test_file():
    global test_file_path
    test_file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if test_file_path:
        test_file_label.config(text=f"Test File: {test_file_path}")
    else:
        test_file_label.config(text="No test file selected")


# 训练模型
def train_model():
    if not train_file_path:
        messagebox.showerror("Error", "Please select a train file first.")
        return

    # 读取训练数据
    data = pd.read_excel(train_file_path)
    X = data.drop(['DATE', 'WEATHER_TYPE'], axis=1)
    y = data['WEATHER_TYPE']

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 定义MLPClassifier并使用GridSearchCV进行超参数调优
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

    # 显示训练进度
    progress_label.config(text="Training model...")

    # 捕获标准输出并显示在 result_text 中
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        grid_search.fit(X_train, y_train)

    # 获取捕获的输出
    captured_output = output_buffer.getvalue()

    # 输出最佳参数
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    # 使用最佳参数训练模型
    best_model = grid_search.best_estimator_

    # 保存模型和标准化器
    joblib.dump(best_model, best_model_path)
    joblib.dump(scaler, scaler_path)

    # 更新 result_text
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Training Log:\n")
    result_text.insert(tk.END, captured_output)
    result_text.insert(tk.END, "\nBest parameters found: {}\n".format(best_params))
    result_text.insert(tk.END, "Model training completed. Best model saved.")

    # 显示训练完成
    progress_label.config(text="Model training completed. Best model saved.")

# 测试模型
def test_model():
    if not test_file_path:
        messagebox.showerror("Error", "Please select a test file first.")
        return

    if not best_model_path or not scaler_path:
        messagebox.showerror("Error", "Please train the model first.")
        return

    # 读取测试数据
    test_data = pd.read_excel(test_file_path)
    X_test = test_data.drop(['DATE', 'WEATHER_TYPE'], axis=1)
    y_test = test_data['WEATHER_TYPE']

    # 加载标准化器
    scaler = joblib.load(scaler_path)
    X_test = scaler.transform(X_test)

    # 加载最优模型
    best_model = joblib.load(best_model_path)

    # 预测并评估模型
    predictions = best_model.predict(X_test)

    # 输出分类报告和混淆矩阵
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    # 显示结果
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Classification Report:\n")
    result_text.insert(tk.END, report)
    result_text.insert(tk.END, "\nConfusion Matrix:\n")
    result_text.insert(tk.END, str(matrix))


# UI布局
# 使用 Frame 将控件分组
file_frame = ttk.Frame(root, padding=10)
file_frame.grid(row=0, column=0, sticky='nsew')

button_frame = ttk.Frame(root, padding=10)
button_frame.grid(row=1, column=0, sticky='nsew')

output_frame = ttk.Frame(root, padding=10)
output_frame.grid(row=2, column=0, sticky='nsew')

# 文件选择部分
train_file_button = ttk.Button(file_frame, text="Select Train File", command=select_train_file)
train_file_button.grid(row=0, column=0, padx=5, pady=5, sticky='w')

train_file_label = ttk.Label(file_frame, text="No train file selected")
train_file_label.grid(row=0, column=1, padx=5, pady=5, sticky='w')

test_file_button = ttk.Button(file_frame, text="Select Test File", command=select_test_file)
test_file_button.grid(row=1, column=0, padx=5, pady=5, sticky='w')

test_file_label = ttk.Label(file_frame, text="No test file selected")
test_file_label.grid(row=1, column=1, padx=5, pady=5, sticky='w')

# 按钮部分
train_button = ttk.Button(button_frame, text="Train Model", command=train_model)
train_button.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

test_button = ttk.Button(button_frame, text="Test Model", command=test_model)
test_button.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

clear_button = ttk.Button(button_frame, text="Clear Log", command=lambda: result_text.delete(1.0, tk.END))
clear_button.grid(row=0, column=2, padx=5, pady=5, sticky='ew')

save_button = ttk.Button(button_frame, text="Save Log", command=lambda: save_log())
save_button.grid(row=0, column=3, padx=5, pady=5, sticky='ew')

# 输出部分
progress_label = ttk.Label(output_frame, text="")
progress_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')

result_text = tk.Text(output_frame, height=20, width=100, wrap='none')
result_text.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

# 添加滚动条
scrollbar_y = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=result_text.yview)
scrollbar_y.grid(row=1, column=1, sticky='ns')
result_text.config(yscrollcommand=scrollbar_y.set)

scrollbar_x = ttk.Scrollbar(output_frame, orient=tk.HORIZONTAL, command=result_text.xview)
scrollbar_x.grid(row=2, column=0, sticky='ew')
result_text.config(xscrollcommand=scrollbar_x.set)

# 保存日志函数
def save_log():
    log_content = result_text.get(1.0, tk.END)
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, "w") as f:
            f.write(log_content)
        messagebox.showinfo("Success", "Log saved successfully.")

# 设置网格权重，使控件能够自动调整大小
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)
output_frame.grid_rowconfigure(1, weight=1)
output_frame.grid_columnconfigure(0, weight=1)

# 运行主循环
root.mainloop()
