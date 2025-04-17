import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys

# add src dir to path cuz imports are broken ughh
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

class PredictiveMaintenanceModel:
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42  # prof said to use 42 lol
        )
        self.svm_model = OneClassSVM(
            kernel='rbf',  # this one works best i tried linear too
            nu=0.1,        # not sure what this does tbh
            gamma='scale'  # auto gves worse results!!
        )
        
    def prepare_ai4i_data(self, df):
        """
        prep the AI4I data for ML stuff
        """
        # features we want (from slides)
        X = df[['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
        
        # make failure types - might b a better way to do this?
        df['Failure Type'] = 'None'
        df.loc[df['TWF'] == 1, 'Failure Type'] = 'TWF'
        df.loc[df['HDF'] == 1, 'Failure Type'] = 'HDF'
        df.loc[df['PWF'] == 1, 'Failure Type'] = 'PWF'  # power failure
        df.loc[df['OSF'] == 1, 'Failure Type'] = 'OSF'
        df.loc[df['RNF'] == 1, 'Failure Type'] = 'RNF'
        
        # only take rows wher failure=1 (duh)
        mask = df['Machine failure'] == 1
        X_failure = X[mask]
        y_failure = df.loc[mask, 'Failure Type']
        
        return X_failure, y_failure
    
    def prepare_battery_data(self, df):
        """
        process battery data for anomaly stuff - this was annoying!!
        """
        # check which columns we hav - diff datasets have diff names :/
        if 'Voltage' in df.columns:
            X = df[['Voltage', 'Current', 'Temperature']]
        else:
            # use actual cols from dataset - these are weird names
            X = df[['c_vol', 'c_cur', 'c_surf_temp']]
        
        # make sure we dont get empty data
        if 'step_type' in df.columns and len(df[df['step_type'] == 'discharge']) > 10:
            # only take discharge steps cuz charge steps are diff pattern
            X = X[df['step_type'] == 'discharge']
    
        # too much data crashes my laptop lol
        if len(X) > 10000:
            X = X.sample(10000, random_state=42)  # random 10k rows
    
        # check we have data!!!!
        if len(X) == 0:
            raise ValueError("no data for battery anomaly detection!!! check filters")
        
        print(f"Using {len(X)} records for anomaly detection")
        return X
    
    def train_failure_classifier(self, X, y):
        """
        train RF for failure prediction - tune params later??
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42  # 80/20 split seems ok
        )
        
        # train model - takes a while on my machine
        self.rf_model.fit(X_train, y_train)
        
        # eval how good it is
        y_pred = self.rf_model.predict(X_test)
        print("\nFailure Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # feature importance - cool to see whats important!
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return self.rf_model
    
    def train_anomaly_detector(self, X):
        """
        train SVM for anomaly detection - harder than classification tbh
        """
        # train model - simpler than RF
        self.svm_model.fit(X)
        
        # get scores to see how anomalous each point is
        scores = self.svm_model.score_samples(X)
        
        # calc threshold - 10% seems ok but maybe try diff values??
        threshold = np.percentile(scores, 10)  # bottom 10% are anomalies
        
        print("\nAnomaly Detection Stats:")
        print(f"Number of samples: {len(X)}")
        print(f"Anomaly threshold score: {threshold:.4f}")
        
        return self.svm_model
    
    def save_models(self, path='models'):
        """
        save models so we dont hav to retrain every time!!!
        """
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.rf_model, os.path.join(path, 'failure_classifier.joblib'))
        joblib.dump(self.svm_model, os.path.join(path, 'anomaly_detector.joblib'))
        print(f"Models saved to: {path}")

if __name__ == "__main__":
    # import our preproc stuff - fix paths
    from preprocessing.process_data import DataPreprocessor
    
    try:
        preprocessor = DataPreprocessor()
        
        # make model obj
        model = PredictiveMaintenanceModel()
        
        # figure out root dir - paths r confusing :/
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Train AI4I failure classifier
        print("Loading AI4I dataset...")
        ai4i_data = preprocessor.load_ai4i_data(os.path.join(base_dir, "ai4i2020.csv"))
        X_ai4i, y_ai4i = model.prepare_ai4i_data(ai4i_data)
        print("Training failure classifier...")
        model.train_failure_classifier(X_ai4i, y_ai4i)
        
        # Train Battery anomaly detector
        print("Loading Battery dataset...")
        battery_data = preprocessor.load_battery_data(os.path.join(base_dir, "Battery Dataset.csv"))
        X_battery = model.prepare_battery_data(battery_data)
        print("Training anomaly detector...")
        model.train_anomaly_detector(X_battery)
        
        # Save models - TODO: add versioning??
        print("Saving models...")
        model.save_models(os.path.join(base_dir, "models"))
        print("Training complete! finally!!")
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1) 