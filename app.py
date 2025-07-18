from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def create_and_train_model():
    """Create and train the model if pkl files don't exist"""
    print("Creating and training new model...")
    
    # Generate synthetic dataset
    np.random.seed(42)
    n_samples = 2000
    
    # Generate features
    age = np.random.normal(55, 15, n_samples).astype(int)
    age = np.clip(age, 20, 90)
    
    sex = np.random.binomial(1, 0.6, n_samples)
    heart_rate = np.random.normal(75, 12, n_samples).astype(int)
    heart_rate = np.clip(heart_rate, 50, 120)
    
    diabetes = np.random.binomial(1, 0.15, n_samples)
    smoking = np.random.binomial(1, 0.25, n_samples)
    alcohol = np.random.binomial(1, 0.4, n_samples)
    previous_heart_problems = np.random.binomial(1, 0.1, n_samples)
    
    bmi = np.random.normal(26, 4, n_samples)
    bmi = np.clip(bmi, 15, 45)
    
    # Create risk score
    risk_score = (
        (age - 20) * 0.02 +
        sex * 0.3 +
        (heart_rate - 60) * 0.01 +
        diabetes * 0.4 +
        smoking * 0.5 +
        alcohol * 0.2 +
        previous_heart_problems * 0.8 +
        (bmi - 18.5) * 0.03 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    heart_attack_risk = (risk_score > np.percentile(risk_score, 70)).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': age,
        'Sex': sex,
        'Heart_Rate': heart_rate,
        'Diabetes': diabetes,
        'Smoking': smoking,
        'Alcohol_Consumption': alcohol,
        'Previous_Heart_Problems': previous_heart_problems,
        'BMI': bmi,
        'Heart_Attack_Risk': heart_attack_risk
    })
    
    # Prepare features and target
    X = data.drop('Heart_Attack_Risk', axis=1)
    y = data['Heart_Attack_Risk']
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, 'heart_attack_model.pkl')
    print("Model trained and saved successfully!")
    
    return model

def load_model():
    """Load the trained model"""
    global model
    
    if os.path.exists('heart_attack_model.pkl'):
        print("Loading existing model...")
        model = joblib.load('heart_attack_model.pkl')
    else:
        print("Model file not found. Creating new model...")
        model = create_and_train_model()
    
    return model

# Load model on startup
load_model()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Get form data
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        heart_rate = float(request.form['heart_rate'])
        diabetes = int(request.form['diabetes'])
        smoking = int(request.form['smoking'])
        alcohol = int(request.form['alcohol'])
        previous_heart = int(request.form['previous_heart'])
        bmi = float(request.form['bmi'])
        
        # Input validation
        if not (20 <= age <= 90):
            return render_template('predict.html', error="Age must be between 20 and 90")
        if not (50 <= heart_rate <= 120):
            return render_template('predict.html', error="Heart rate must be between 50 and 120")
        if not (15 <= bmi <= 45):
            return render_template('predict.html', error="BMI must be between 15 and 45")
        
        # Make prediction
        input_data = np.array([[age, sex, heart_rate, diabetes, smoking, alcohol, previous_heart, bmi]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Prepare result
        risk_level = "High" if prediction == 1 else "Low"
        risk_percentage = probability * 100
        
        # Risk category
        if risk_percentage < 30:
            risk_category = "Low Risk"
            risk_color = "success"
        elif risk_percentage < 60:
            risk_category = "Moderate Risk"
            risk_color = "warning"
        else:
            risk_category = "High Risk"
            risk_color = "danger"
        
        # Generate recommendations
        recommendations = generate_recommendations(age, sex, heart_rate, diabetes, smoking, alcohol, previous_heart, bmi)
        
        return render_template('predict.html', 
                             prediction=True,
                             risk_level=risk_level,
                             risk_percentage=round(risk_percentage, 2),
                             risk_category=risk_category,
                             risk_color=risk_color,
                             recommendations=recommendations,
                             input_data={
                                 'age': age,
                                 'sex': 'Male' if sex == 1 else 'Female',
                                 'heart_rate': heart_rate,
                                 'diabetes': 'Yes' if diabetes == 1 else 'No',
                                 'smoking': 'Yes' if smoking == 1 else 'No',
                                 'alcohol': 'Yes' if alcohol == 1 else 'No',
                                 'previous_heart': 'Yes' if previous_heart == 1 else 'No',
                                 'bmi': bmi
                             })
        
    except Exception as e:
        return render_template('predict.html', error=f"Error making prediction: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Extract features
        features = [
            data['age'],
            data['sex'],
            data['heart_rate'],
            data['diabetes'],
            data['smoking'],
            data['alcohol'],
            data['previous_heart'],
            data['bmi']
        ]
        
        # Make prediction
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High' if prediction == 1 else 'Low',
            'risk_percentage': round(probability * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/privacy')
def privacy():
    """Privacy policy page"""
    return render_template('privacy.html')

def generate_recommendations(age, sex, heart_rate, diabetes, smoking, alcohol, previous_heart, bmi):
    """Generate personalized health recommendations"""
    recommendations = []
    
    # Age-based recommendations
    if age > 60:
        recommendations.append("Regular cardiac check-ups are recommended for people over 60")
    
    # BMI recommendations
    if bmi > 30:
        recommendations.append("Consider weight management - BMI over 30 indicates obesity")
    elif bmi > 25:
        recommendations.append("Maintain healthy weight - BMI over 25 indicates overweight")
    
    # Heart rate recommendations
    if heart_rate > 100:
        recommendations.append("High resting heart rate detected - consult with a cardiologist")
    elif heart_rate < 60:
        recommendations.append("Low resting heart rate - monitor during physical activity")
    
    # Lifestyle recommendations
    if smoking:
        recommendations.append("Quit smoking immediately - it's the most important step for heart health")
    
    if alcohol:
        recommendations.append("Limit alcohol consumption to moderate levels")
    
    if diabetes:
        recommendations.append("Keep diabetes well-controlled with proper medication and diet")
    
    if previous_heart:
        recommendations.append("Follow up regularly with your cardiologist for ongoing heart conditions")
    
    # General recommendations
    recommendations.extend([
        "Maintain a healthy diet rich in fruits, vegetables, and whole grains",
        "Exercise regularly - aim for 150 minutes of moderate exercise per week",
        "Manage stress through relaxation techniques or meditation",
        "Get adequate sleep (7-9 hours per night)",
        "Monitor blood pressure regularly"
    ])
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)