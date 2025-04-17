# Predictive Maintenance of Machine Components using Machine Learning

## Project Overview
This project implements a machine learning-based predictive maintenance system for industrial machine components. It focuses on analyzing sensor data (vibration, temperature, voltage) to predict component failures and optimize maintenance schedules.

## Features
- Signal processing of vibration data using FFT and Wavelet Transforms
- Time and frequency domain feature extraction
- Multi-class failure classification using Random Forest
- Anomaly detection using SVM with RBF Kernel
- Interactive visualization dashboard
- Real-time maintenance scheduling

## Dataset
The project uses two main datasets:
1. AI4I 2020 Predictive Maintenance Dataset
   - Features: Tool wear, torque, temperature, rotational speed
   - Failure modes: TWF (Tool Wear Failure), HDF (Heat Dissipation Failure), PWF (Power Failure), OSF (Overstrain Failure), RNF (Random Failure)

2. Battery Dataset
   - Features: Voltage, current, temperature, cycle life

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup
1. Clone the repository
2. Create a virtual environment:
   ```powershell
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

The project includes a convenient run script that handles both training and running the web dashboard:

### Training the models
```powershell
python run.py --train
```

This command will:
1. Load and preprocess the datasets
2. Train the Random Forest classifier for failure prediction
3. Train the One-Class SVM for anomaly detection
4. Save the trained models

### Starting the web dashboard
```powershell
python run.py --start
```

This command will start the Flask web application at http://127.0.0.1:5000, where you can:
- Predict failure types based on machine parameters
- Detect anomalies in battery data
- View feature importance visualizations

### Train and start in one command
```powershell
python run.py --train --start
```

## Project Structure
```
├── data/                  # Dataset storage
├── src/                   # Source code
│   ├── preprocessing/     # Data preprocessing modules
│   └── models/           # ML model implementations
├── notebooks/            # Jupyter notebooks for analysis
├── web_dashboard/        # Flask web application
└── models/               # Saved model files
```

## Technical Implementation

### 1. Data Preprocessing
- **Signal Processing**: Applying FFT for frequency-domain analysis and Wavelet Transforms for multi-resolution time-frequency analysis.
- **Feature Engineering**: Extracting time-domain features (RMS, Kurtosis, Crest Factor) and frequency-domain features (Dominant frequencies, spectral entropy).

### 2. Model Development
- **Random Forest Classifier**: Used for multi-class failure classification, providing feature importance rankings.
- **One-Class SVM**: Deployed for anomaly detection in battery data, using an RBF kernel to capture non-linear relationships.

### 3. Web Dashboard
- **Interactive UI**: Real-time prediction interface with visual feedback.
- **Feature Importance**: Interactive plots showing the contribution of each feature to failure prediction.

## Results
The system provides:
- Failure prediction with >90% accuracy
- Real-time anomaly detection for battery systems
- Maintenance schedule optimization
- Interactive visualizations of component health

## Future Improvements
- Integration with IoT sensors for real-time monitoring
- Development of a remaining useful life (RUL) prediction model
- Incorporation of deep learning models for complex pattern recognition

## License
MIT License

## Author
[Rishabh Vats]

## Acknowledgments
- AI4I 2020 Predictive Maintenance Dataset
- Battery Dataset contributors 