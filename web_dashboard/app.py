from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.utils
import json
import sys
import numpy as np

app = Flask(__name__)

# vars for models
rf_model = None
svm_model = None
models_loaded = False

# try 2 load models if they exist - sometimes they dont!!
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

try:
    if os.path.exists(os.path.join(MODEL_PATH, 'failure_classifier.joblib')):
        rf_model = joblib.load(os.path.join(MODEL_PATH, 'failure_classifier.joblib'))
        svm_model = joblib.load(os.path.join(MODEL_PATH, 'anomaly_detector.joblib'))
        models_loaded = True
        print("Models loaded successfully!")
    else:
        print("Model files not found. You need to train models first with 'python src/models/train.py'")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_failure', methods=['POST'])
def predict_failure():
    if not models_loaded or rf_model is None:
        return jsonify({
            'error': 'Models not loaded. Please train the models first.'
        }), 503
    
    data = request.json
    features = pd.DataFrame([data])
    
    # predict failures - main functionality!!
    prediction = rf_model.predict(features)[0]
    probabilities = rf_model.predict_proba(features)[0]
    
    return jsonify({
        'prediction': prediction,
        'probabilities': probabilities.tolist()
    })

@app.route('/detect_anomaly', methods=['POST'])
def detect_anomaly():
    if not models_loaded or svm_model is None:
        return jsonify({
            'error': 'Models not loaded. Please train the models first.'
        }), 503
    
    try:
        data = request.json
        
        # map frontend names to model names
        mapped_data = {
            'c_vol': data['Voltage'],
            'c_cur': data['Current'],
            'c_surf_temp': data['Temperature']
        }
        
        features = pd.DataFrame([mapped_data])
        
        # define normal ranges - based on lithium battery specs
        normal_ranges = {
            'c_vol': (2.5, 4.5),  # typical volts for lithium bat
            'c_cur': (0.1, 5),     # normal current range
            'c_surf_temp': (15, 60)  # safe temp range - over 60 = danger!!
        }
        
        # do range checks first - catch obvious problems
        obvious_anomaly = False
        anomaly_reason = []
        
        for feature, (min_val, max_val) in normal_ranges.items():
            if features[feature].iloc[0] < min_val or features[feature].iloc[0] > max_val:
                obvious_anomaly = True
                anomaly_reason.append(f"{feature.replace('c_', '').capitalize()} out of normal range")
        
        # also check model score - catches subtle issues
        raw_score = svm_model.score_samples(features)[0]
        
        # svm gives negative scores for anomalies
        threshold = -0.01  # found this by trial&error
        
        # normalize score to 0-10 scale cuz its easier to understand
        # had to tinker with the formula a lot to get good results!
        normalized_score = max(0, 10 + raw_score * 20)
        
        # reduce score if there are range violations
        if obvious_anomaly:
            # how much out of range is it?? further = worse
            severity = 0
            for feature, (min_val, max_val) in normal_ranges.items():
                val = features[feature].iloc[0]
                if val < min_val:
                    # calc how far below min
                    severity += min(1.0, (min_val - val) / min_val)
                elif val > max_val:
                    # calc how far above max
                    severity += min(1.0, (val - max_val) / max_val)
            
            # reduce score based on severity
            normalized_score = max(0, normalized_score - severity * 5)
        
        # final decision
        model_detected_anomaly = raw_score < threshold
        is_anomaly = obvious_anomaly or model_detected_anomaly
        
        # create response w/ details
        response = {
            'score': round(normalized_score, 4),
            'raw_score': float(raw_score),
            'is_anomaly': bool(is_anomaly),
            'anomaly_reason': anomaly_reason if obvious_anomaly else 
                              ['Unusual pattern detected by model'] if model_detected_anomaly else []
        }
        
        app.logger.debug(f"Anomaly detection - Data: {mapped_data}, Score: {raw_score}, Normalized: {normalized_score}, Is Anomaly: {is_anomaly}")
        
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error in anomaly detection: {str(e)}")
        return jsonify({
            'error': f"Error processing request: {str(e)}"
        }), 500

@app.route('/feature_importance')
def feature_importance():
    if not models_loaded or rf_model is None:
        # return fake data if model isnt loaded
        dummy_data = {
            'data': [{
                'x': ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear'],
                'y': [0.2, 0.25, 0.15, 0.3, 0.1],
                'type': 'bar'
            }],
            'layout': {
                'title': 'Feature Importance (Example Data - Model Not Trained)'
            }
        }
        return jsonify(dummy_data)
    
    # get importance from RF
    importance = pd.DataFrame({
        'feature': ['Air temperature', 'Process temperature', 
                   'Rotational speed', 'Torque', 'Tool wear'],
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # make bar chart - plotly is better than matplotlib
    fig = px.bar(importance, x='feature', y='importance',
                 title='Feature Importance for Failure Prediction')
    
    return jsonify(json.loads(fig.to_json()))

if __name__ == '__main__':
    app.run(debug=True)  # turn debug off for prod!! 