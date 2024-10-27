from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load pre-trained scaler and model
scaler = pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Capture form data and preprocess it
        input_data = [
            int(request.form.get("Pregnancies")),
            float(request.form.get('Glucose')),
            float(request.form.get('BloodPressure')),
            float(request.form.get('SkinThickness')),
            float(request.form.get('Insulin')),
            float(request.form.get('BMI')),
            float(request.form.get('DiabetesPedigreeFunction')),
            float(request.form.get('Age'))
        ]
        
        # Scale the input data
        scaled_data = scaler.transform([input_data])
        prediction = model.predict(scaled_data)
        
        # Prediction label
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        
        return render_template('single_prediction.html', result=result)
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
