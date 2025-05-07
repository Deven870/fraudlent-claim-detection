# Healthcare Fraud Detection System

A modern web-based application for detecting potentially fraudulent healthcare claims using AI and machine learning.

![Healthcare Fraud Detection System](https://via.placeholder.com/800x400?text=Healthcare+Fraud+Detection+Interface)

## Overview

This system uses a K-Nearest Neighbors model trained on healthcare claim data to identify potentially fraudulent claims. The application features:

- Modern, responsive web interface for entering claim details
- Real-time prediction using a pre-trained machine learning model
- Visual representation of model performance metrics
- Interactive data visualization with animations
- Detailed results with probability scores and visual analysis
- Feature importance visualization

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
```
git clone <repository-url>
cd <repository-name>
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

3. Ensure model files are present:
   - The application requires two trained model files:
     - `knn_fraud_model.pkl`: The trained KNN model
     - `scaler.pkl`: The StandardScaler used for feature normalization
   - These should be placed in the root directory of the application

### Running the Application

1. Start the Flask server:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

## Usage

1. Enter claim details in the form on the home page
2. Click "Predict Fraud" to submit
3. View the prediction results, including:
   - Fraud/No Fraud classification
   - Probability score visualization
   - Visual representation of the key features
   - Summary of input features with icons

## Input Features

The model requires 10 specific features for prediction:

| Feature | Description |
|---------|-------------|
| Renal Disease Indicator | Binary indicator (0 or 1) of renal disease |
| Claim Amount Reimbursed | Amount that was reimbursed for this claim |
| IP Annual Reimbursement | Inpatient annual reimbursement amount |
| OP Annual Reimbursement | Outpatient annual reimbursement amount |
| Deductible Amount Paid | Deductible amount paid for the claim |
| IP Annual Deductible | Inpatient annual deductible amount |
| OP Annual Deductible | Outpatient annual deductible amount |
| Length of Stay | Duration of hospital stay in days |
| Claim Duration | Duration of the claim in days |
| Patient Age | Patient's age in years |

## API Usage

The system also provides a REST API endpoint for integration with other systems:

```
POST /api/predict
```

Example request body:
```json
{
  "renaldiseaseindicator": 1,
  "dr_inscclaimamtreimbursed": 6084.00,
  "dr_ipannualreimbursementamt": 12000.00,
  "dr_opannualreimbursementamt": 5000.00,
  "dr_deductibleamtpaid": 500.00,
  "dr_ipannualdeductibleamt": 1000.00,
  "dr_opannualdeductibleamt": 500.00,
  "dr_los": 5.0,
  "dr_duration": 14.0,
  "dr_age": 65.0
}
```

Example response:
```json
{
  "fraud": 1,
  "probability": 0.85,
  "features_used": {
    "renaldiseaseindicator": 1,
    "dr_inscclaimamtreimbursed": 6084.0,
    ...
  }
}
```

## Features

- **Modern UI**: Clean, responsive interface with intuitive form controls
- **Visual Feedback**: Charts and visual elements to help understand model results
- **Interactive Elements**: Animations and hover effects for better user experience
- **Mobile Responsive**: Works on devices of all sizes
- **API Integration**: REST API for integration with other systems

## Technologies Used

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Visualization**: Chart.js
- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn (K-Nearest Neighbors)
- **Animation**: AOS (Animate On Scroll)
- **Icons**: Boxicons

## License

This project is licensed under the MIT License - see the LICENSE file for details. 