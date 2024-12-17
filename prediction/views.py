import joblib
import numpy as np
from django.shortcuts import render

# Load model and scaler
model = joblib.load('prediction/models/insurance_model.joblib')
scaler = joblib.load('prediction/models/scaler.joblib')

def predict_charges(request):
    if request.method == 'POST':
        # Extract inputs from the POST request
        age = float(request.POST.get('age'))
        sex = int(request.POST.get('sex'))  # 0: Female, 1: Male
        bmi = float(request.POST.get('bmi'))
        children = int(request.POST.get('children'))
        smoker = int(request.POST.get('smoker'))  # 0: Non-smoker, 1: Smoker
        region = int(request.POST.get('region'))  # Encoded regions

        # Scale numerical features
        scaled_features = scaler.transform([[age, bmi, children]])

        # Combine scaled features with categorical inputs
        inputs = np.hstack([scaled_features, np.array([sex, smoker, region]).reshape(1, -1)])

        # Make prediction
        prediction = model.predict(inputs)[0]

        # Render prediction result
        return render(request, 'prediction/result.html', {'prediction': round(prediction, 2)})

    # Render the form for input
    return render(request, 'prediction/index.html')
def home(request):
    return render(request, 'prediction/home.html')