import configparser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def read_ini_data(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    data = []
    for section in config.sections():
        single_data = {key: float(val) for key, val in config.items(section)}
        data.append(single_data)
    return pd.DataFrame(data)

# 示例文件路径，替换为您的实际文件路径
file_path = 'D:\\shudeng\\ProofingTool\\数据\\example.ini'

# 读取数据
df = read_ini_data(file_path)

# 假设您的标签列是名为 'Target' 的列
X = df.drop('Target', axis=1)
y = df['Target']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# SVM模型
svm_model = SVC()
svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(svm_model, svm_param_grid, cv=5)
svm_grid.fit(X_train, y_train)
svm_pred = svm_grid.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

# 决策树模型
tree_model = DecisionTreeClassifier()
tree_param_grid = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
tree_grid = GridSearchCV(tree_model, tree_param_grid, cv=5)
tree_grid.fit(X_train, y_train)
tree_pred = tree_grid.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)

# 随机森林模型
rf_model = RandomForestClassifier()
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5)
rf_grid.fit(X_train, y_train)
rf_pred = rf_grid.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# 比较模型结果
print("SVM Accuracy:", svm_accuracy)
print("Decision Tree Accuracy:", tree_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

