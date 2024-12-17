import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from .models import InsurancePrediction
from django.shortcuts import render
scaler = StandardScaler()
from django.db.models import Avg, Count, Sum
from django.utils.timezone import now
import datetime
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

        # Save the data to the database
        prediction_data = InsurancePrediction(
            age=age,
            sex=sex,
            bmi=bmi,
            children=children,
            smoker=smoker,
            region=region,
            predicted_charge=round(prediction, 2)
        )
        prediction_data.save()

        # Render prediction result
        return render(request, 'prediction/result.html', {'prediction': round(prediction, 2)})

    # Render the form for input
    return render(request, 'prediction/index.html')
def home(request):
    return render(request, 'prediction/home.html')

def dashboard(request):
    # Fetching necessary data
    total_predictions = InsurancePrediction.objects.count()
    average_charge = InsurancePrediction.objects.aggregate(Avg('predicted_charge'))['predicted_charge__avg']
    
    # Group by smoker status (0: Non-smoker, 1: Smoker)
    predictions_by_smoker = InsurancePrediction.objects.values('smoker').annotate(total=Count('id'))
    smoker_labels = ['Non-smoker', 'Smoker']
    smoker_data = [0, 0]
    for prediction in predictions_by_smoker:
        smoker_data[prediction['smoker']] = prediction['total']

    # Group by region
    predictions_by_region = InsurancePrediction.objects.values('region').annotate(total=Count('id'))
    region_labels = ['North', 'South', 'East', 'West']  # Adjust based on your region IDs
    region_data = [0, 0, 0, 0]
    for prediction in predictions_by_region:
        region_data[prediction['region']] = prediction['total']

    # Fetch predictions for scatter plot (Age vs Predicted Charge)
    predictions_for_chart = InsurancePrediction.objects.values('age', 'predicted_charge')
    scatter_data = [{'x': pred['age'], 'y': pred['predicted_charge']} for pred in predictions_for_chart]

    # Passing data to the template
    context = {
        'total_predictions': total_predictions,
        'average_charge': average_charge,
        'smoker_labels': smoker_labels,
        'smoker_data': smoker_data,
        'region_labels': region_labels,
        'region_data': region_data,
        'scatter_data': scatter_data,
    }

    return render(request, 'prediction/dashboard.html', context)