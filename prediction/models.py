from django.db import models

class InsurancePrediction(models.Model):
    age = models.FloatField()
    sex = models.IntegerField(choices=[(0, 'Female'), (1, 'Male')])
    bmi = models.FloatField()
    children = models.IntegerField()
    smoker = models.IntegerField(choices=[(0, 'Non-smoker'), (1, 'Smoker')])
    region = models.IntegerField()
    predicted_charge = models.FloatField()
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction on {self.date_created} for Age {self.age}"
