from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        radius = float(request.form['radius'])
        texture = float(request.form['texture'])
        perimeter = float(request.form['perimeter'])
        area = float(request.form['area'])
        concavity = float(request.form['concavity'])
        concave_points = float(request.form['concave_points'])

        features = np.array([[radius, texture, perimeter, area, concavity, concave_points]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        if prediction == 1:
            result = f"Malignant (High Risk) - Probability: {probability*100:.2f}%"
        else:
            result = f"Benign (Low Risk) - Probability: {(1-probability)*100:.2f}%"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)