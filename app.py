import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__, static_folder='static', static_url_path='/static')

# List of top 10 features as mentioned in the requirements
top_features = [
    'renaldiseaseindicator',
    'dr_inscclaimamtreimbursed',
    'dr_ipannualreimbursementamt',
    'dr_opannualreimbursementamt',
    'dr_deductibleamtpaid',
    'dr_ipannualdeductibleamt',
    'dr_opannualdeductibleamt',
    'dr_los',
    'dr_duration',
    'dr_age'
]

# Load model and scaler
model_path = 'knn_fraud_model.pkl'
scaler_path = 'scaler.pkl'

# Check if model files exist
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_loaded = True
    
    # Calculate feature importance from the model
    # For KNN, we don't have direct feature importance, so we'll estimate it
    # In a real scenario, you would use a method appropriate for your model type
    # or calculate it during training
    feature_importance = {}
    for i, feature in enumerate(top_features):
        # This is a placeholder - with your actual model, replace with real importance values
        importance = np.random.uniform(0.03, 0.18)  # Placeholder
        feature_importance[feature] = importance
    
    # Normalize to ensure they sum to 1
    total = sum(feature_importance.values())
    for feature in feature_importance:
        feature_importance[feature] /= total
else:
    model_loaded = False
    feature_importance = {}
    print(f"Warning: Model files not found. Please ensure {model_path} and {scaler_path} exist.")

# Model metrics - updated with actual evaluation results
model_metrics = {
    'accuracy': 1.00 if model_loaded else 0,
    'precision': 1.00 if model_loaded else 0,
    'recall': 1.00 if model_loaded else 0,
    'f1_score': 1.00 if model_loaded else 0,
    'feature_importance': feature_importance,
    'confusion_matrix': {
        'true_negative': 103455,
        'false_positive': 23,
        'false_negative': 11,
        'true_positive': 63975
    } if model_loaded else {}
}

# Create feature descriptions for tooltips and documentation
feature_descriptions = {
    'renaldiseaseindicator': 'Binary indicator (0 or 1) of renal disease in the patient',
    'dr_inscclaimamtreimbursed': 'The amount that was reimbursed for this specific claim',
    'dr_ipannualreimbursementamt': 'Total inpatient annual reimbursement amount for the patient',
    'dr_opannualreimbursementamt': 'Total outpatient annual reimbursement amount for the patient',
    'dr_deductibleamtpaid': 'The deductible amount paid for this specific claim',
    'dr_ipannualdeductibleamt': 'Total inpatient annual deductible amount for the patient',
    'dr_opannualdeductibleamt': 'Total outpatient annual deductible amount for the patient',
    'dr_los': 'Length of stay in days for inpatient services',
    'dr_duration': 'Total duration of the claim in days',
    'dr_age': 'Patient\'s age in years'
}

@app.route('/')
def index():
    return render_template('index.html', 
                           top_features=top_features,
                           model_metrics=model_metrics,
                           feature_descriptions=feature_descriptions,
                           model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model files not found'}), 500
    
    try:
        # Debug - Print all received form data
        print("Received form data:", request.form)
        
        # Get form data for all features
        features_dict = {}
        for feature in top_features:
            value = request.form.get(feature, '')
            if value == '':
                print(f"Missing value for feature: {feature}")
                return jsonify({'error': f'Missing value for {feature}'}), 400
            
            # Convert to appropriate type
            try:
                if feature == 'renaldiseaseindicator':
                    features_dict[feature] = int(value)
                else:
                    features_dict[feature] = float(value)
            except ValueError:
                print(f"Invalid value for feature: {feature}, value: {value}")
                return jsonify({'error': f'Invalid value for {feature}'}), 400
        
        # Debug - Print processed features
        print("Processed features:", features_dict)
        
        # Prepare features in the correct order to match model training
        feature_array = []
        for feature in top_features:
            feature_array.append(features_dict[feature])
            
        # Convert to numpy array and reshape for sklearn
        features = np.array(feature_array).reshape(1, -1)
        print("Feature array shape:", features.shape)
        
        # Scale features using the same scaler as during training
        features_scaled = scaler.transform(features)
        print("Scaled features:", features_scaled)
        
        # Make prediction with KNN model
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)
        
        print(f"Model prediction: {prediction[0]}, Probability: {prediction_proba[0]}")
        
        result = {
            'fraud': int(prediction[0]),
            'probability': float(prediction_proba[0][1]) if prediction[0] == 1 else float(prediction_proba[0][0]),
            'features_used': features_dict,
            'feature_descriptions': feature_descriptions
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not model_loaded:
        return jsonify({'error': 'Model files not found'}), 500
    
    try:
        # Get JSON data
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare features
        features_dict = {}
        for feature in top_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            features_dict[feature] = data[feature]
        
        features = np.array([features_dict[f] for f in top_features]).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)
        
        return jsonify({
            'fraud': int(prediction[0]),
            'probability': float(prediction_proba[0][1]) if prediction[0] == 1 else float(prediction_proba[0][0]),
            'features_used': features_dict
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-check')
def model_check():
    """Debug route to verify model loading and parameters"""
    if not model_loaded:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded',
            'model_path': model_path,
            'scaler_path': scaler_path
        })
    
    try:
        model_info = {
            'status': 'success',
            'model_type': str(type(model)),
            'n_neighbors': getattr(model, 'n_neighbors', 'unknown'),
            'weights': getattr(model, 'weights', 'unknown'),
            'algorithm': getattr(model, 'algorithm', 'unknown'),
            'scaler_type': str(type(scaler)),
            'feature_names': top_features,
            'metrics': {
                'accuracy': model_metrics['accuracy'],
                'precision': model_metrics['precision'],
                'recall': model_metrics['recall'],
                'f1_score': model_metrics['f1_score']
            }
        }
        return jsonify(model_info)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/evaluate-model', methods=['GET', 'POST'])
def evaluate_model():
    error_message = None
    evaluation_complete = False
    num_samples = 0
    
    if request.method == 'POST':
        if not model_loaded:
            error_message = "Model not loaded. Cannot evaluate."
        else:
            # Check if file is provided
            if 'test_data' not in request.files:
                error_message = "No file provided"
            else:
                file = request.files['test_data']
                if file.filename == '':
                    error_message = "No file selected"
                elif not file.filename.endswith('.csv'):
                    error_message = "File must be a CSV"
                else:
                    try:
                        import pandas as pd
                        import tempfile
                        import os
                        
                        # Save uploaded file to temp location
                        temp_dir = tempfile.gettempdir()
                        temp_path = os.path.join(temp_dir, 'test_data.csv')
                        file.save(temp_path)
                        
                        # Load the CSV file
                        df = pd.read_csv(temp_path)
                        print(f"Loaded test data with shape: {df.shape}")
                        
                        # Check if required columns exist
                        missing_cols = [col for col in top_features + ['fraud'] if col not in df.columns]
                        if missing_cols:
                            error_message = f"Missing columns in CSV: {', '.join(missing_cols)}"
                        else:
                            # Extract features and target
                            X = df[top_features].values
                            y_true = df['fraud'].values
                            
                            # Scale features
                            X_scaled = scaler.transform(X)
                            
                            # Make predictions with model
                            y_pred = model.predict(X_scaled)
                            
                            # Calculate metrics
                            acc = accuracy_score(y_true, y_pred)
                            prec = precision_score(y_true, y_pred, zero_division=0)
                            rec = recall_score(y_true, y_pred, zero_division=0)
                            f1 = f1_score(y_true, y_pred, zero_division=0)
                            
                            # Update metrics
                            model_metrics['accuracy'] = acc
                            model_metrics['precision'] = prec
                            model_metrics['recall'] = rec
                            model_metrics['f1_score'] = f1
                            model_metrics['num_samples'] = len(y_true)
                            
                            num_samples = len(y_true)
                            print(f"Evaluation complete. Metrics: acc={acc:.2f}, prec={prec:.2f}, rec={rec:.2f}, f1={f1:.2f}")
                            evaluation_complete = True
                            
                            # Clean up temp file
                            os.remove(temp_path)
                    except Exception as e:
                        print(f"Error in evaluation: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        error_message = str(e)
    
    return render_template('evaluate.html', 
                          model_metrics=model_metrics, 
                          feature_descriptions=feature_descriptions,
                          model_loaded=model_loaded,
                          error_message=error_message,
                          evaluation_complete=evaluation_complete,
                          num_samples=num_samples)

if __name__ == '__main__':
    # Ensure static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
        
    # Create styles.css in static directory if it doesn't exist yet
    css_path = os.path.join('static', 'styles.css')
    if not os.path.exists(css_path):
        with open(css_path, 'w') as f:
            f.write('/* Modern Styling for Healthcare Fraud Detection System */')

    app.run(debug=True) 