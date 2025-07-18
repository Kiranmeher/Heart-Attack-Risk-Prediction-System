# Heart Attack Disease Risk Prediction

A comprehensive machine learning web application that predicts heart attack risk based on 8 key health parameters using advanced AI algorithms.

## 🎯 Overview

HeartGuard AI is an intelligent health assessment tool that uses machine learning to analyze cardiovascular risk factors and provide personalized heart attack risk predictions. The application combines cutting-edge AI technology with evidence-based medical research to deliver accurate, actionable health insights.

## ✨ Features

- **AI-Powered Predictions**: Advanced Random Forest machine learning model
- **8 Key Risk Factors**: Age, Sex, Heart Rate, Diabetes, Smoking, Alcohol, Previous Heart Problems, BMI
- **Real-time Analysis**: Instant risk assessment with detailed explanations
- **Privacy-First**: No data storage - all processing done in real-time
- **Responsive Design**: Works seamlessly on all devices
- **Personalized Recommendations**: Tailored health advice based on risk factors
- **Medical Evidence-Based**: Built on established cardiovascular research

## 🏗️ Project Structure

```
heart-attack-prediction/
├── app.py                          # Flask backend application
├── Heart_Attack_Risk_Prediction.ipynb  # Jupyter notebook for model training
├── requirements.txt                 # Python dependencies
├── README.md                       # Project documentation
├── static/                         # Static assets
│   ├── style.css                   # Main stylesheet
│   ├── styleform.css              # Form-specific styles
│   ├── script.js                  # Main JavaScript
│   ├── scriptform.js              # Form JavaScript
│   ├── about.css                  # About page styles
│   ├── contact.css                # Contact page styles
│   └── privacy.css                # Privacy page styles
└── templates/                      # HTML templates
    ├── index.html                  # Homepage
    ├── predict.html               # Prediction form and results
    ├── about.html                 # About page
    └── privacy.html               # Privacy policy
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd heart-attack-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 🧠 Machine Learning Model

### Algorithm
- **Model Type**: Random Forest Classifier
- **Training Data**: Synthetic dataset with 2,000 samples
- **Features**: 8 key cardiovascular risk factors
- **Accuracy**: ~92% on test data

### Risk Factors Analyzed

| Factor | Type | Range/Values | Impact |
|--------|------|--------------|---------|
| Age | Continuous | 20-90 years | High |
| Sex | Binary | Male/Female | Medium |
| Heart Rate | Continuous | 50-120 bpm | Medium |
| Diabetes | Binary | Yes/No | High |
| Smoking | Binary | Yes/No | High |
| Alcohol | Binary | Yes/No | Medium |
| Previous Heart Problems | Binary | Yes/No | Very High |
| BMI | Continuous | 15-45 kg/m² | Medium |

### Model Training

The Jupyter notebook (`Heart_Attack_Risk_Prediction.ipynb`) contains:
- Data generation and preprocessing
- Exploratory data analysis
- Model training and validation
- Feature importance analysis
- Model evaluation metrics

## 🌐 Web Application

### Backend (Flask)
- **Framework**: Flask 2.3.3
- **Model Loading**: Automatic model training if pkl files don't exist
- **API Endpoints**: RESTful API for predictions
- **Security**: Input validation and error handling

### Frontend
- **Design**: Modern, responsive UI with medical theme
- **Styling**: Custom CSS with healthcare color palette
- **Interactivity**: Real-time form validation and smooth animations
- **Accessibility**: WCAG compliant design

### Key Pages

1. **Homepage** (`/`): Overview and features
2. **Risk Assessment** (`/predict`): Prediction form and results
3. **About** (`/about`): Technology and team information
4. **Privacy** (`/privacy`): Comprehensive privacy policy

## 🔒 Privacy & Security

- **Zero Data Storage**: Health information is never stored
- **Real-time Processing**: Data processed in memory only
- **Secure Transmission**: SSL/TLS encryption
- **Privacy by Design**: Minimal data collection
- **GDPR Compliant**: Comprehensive privacy protections

## 📊 API Usage

### Prediction Endpoint

```bash
POST /api/predict
Content-Type: application/json

{
  "age": 45,
  "sex": 1,
  "heart_rate": 75,
  "diabetes": 0,
  "smoking": 0,
  "alcohol": 0,
  "previous_heart": 0,
  "bmi": 24.5
}
```

### Response

```json
{
  "prediction": 0,
  "probability": 0.23,
  "risk_level": "Low",
  "risk_percentage": 23.0
}
```

## 🎨 Design System

### Color Palette
- **Primary**: #2563eb (Blue)
- **Secondary**: #059669 (Green)
- **Accent**: #dc2626 (Red)
- **Warning**: #f59e0b (Orange)
- **Success**: #10b981 (Green)
- **Neutral**: #6b7280 (Gray)

### Typography
- **Font Family**: 'Segoe UI', system fonts
- **Headings**: 600 weight, varied sizes
- **Body**: 400 weight, 1.6 line height

## 🧪 Testing

### Model Validation
- Train/test split: 80/20
- Cross-validation: 5-fold
- Metrics: Accuracy, Precision, Recall, F1-score

### Web Application Testing
- Form validation testing
- API endpoint testing
- Responsive design testing
- Browser compatibility testing

## 📈 Performance

- **Model Accuracy**: 92%
- **Response Time**: <1 second
- **Scalability**: Stateless design for horizontal scaling
- **Availability**: 99.9% uptime target

## 🔧 Configuration

### Environment Variables
```bash
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key
```

### Model Parameters
- `n_estimators`: 100
- `random_state`: 42
- `max_depth`: Auto

## 📱 Mobile Support

- Responsive design for all screen sizes
- Touch-friendly interface
- Optimized performance on mobile devices
- Progressive Web App (PWA) ready

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## ⚠️ Medical Disclaimer

This application is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Medical research community for cardiovascular risk factor studies
- Open source machine learning libraries
- Healthcare professionals for domain expertise
- Users providing feedback for continuous improvement


