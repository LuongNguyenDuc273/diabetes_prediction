import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np
# Đọc dữ liệu
data = pd.read_csv('diabetes_prediction_dataset_final.csv')

# Xử lý dữ liệu
# ... (Có thể cần xử lý dữ liệu thiếu, chuyển đổi kiểu dữ liệu, v.v.)

# Tách dữ liệu thành tập train và test
X = data[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = data['diabetes']

kfold = KFold(n_splits=10, shuffle=True, random_state=42)  # Thay thế n-split bằng số lượng tập con `k` mong muốn

scores_accuracy = []
scores_precision = []
scores_recall = []
scores_f1 = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    scores_accuracy.append(accuracy)
    scores_precision.append(precision)
    scores_recall.append(recall)
    scores_f1.append(f1)

print("Điểm trung bình:")
print("Accuracy:", np.mean(scores_accuracy))
print("Precision:", np.mean(scores_precision))
print("Recall:", np.mean(scores_recall))
print("F1 score:", np.mean(scores_f1))

