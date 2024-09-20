# Import thư viện cần thiết
from flask import Flask, render_template, request
import pickle
import numpy as np 
from sklearn.ensemble import RandomForestClassifier

# Load model Random Forest
filename = 'model_RDF.pkl'
classifier = pickle.load(open(filename, 'rb'))
# Chạy ứng dụng web
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hypt = int(request.form['hypertension'])
        hd = int(request.form['heart_disease'])
        smk = int(request.form['smoking_history'])
        bmi = float(request.form['bmi'])
        hba1c = float(request.form['hba1c_level'])
        bgl = int(request.form['blood_glucose_level'])
        
        data = np.array([[gender, age, hypt, hd, smk, bmi, hba1c, bgl]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)