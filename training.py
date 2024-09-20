import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Đọc dữ liệu
data = pd.read_csv('diabetes_prediction_dataset_final.csv')

# Xử lý dữ liệu
# ... (Có thể cần xử lý dữ liệu thiếu, chuyển đổi kiểu dữ liệu, v.v.)

# Tách dữ liệu thành tập train và test
X = data[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = data['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Đánh giá hiệu suất mô hình trên tập test
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("F1 score:", f1)


# Lưu mô hình
# import pickle

# pickle.dump(model, open('model_RDF.pkl', 'wb'))