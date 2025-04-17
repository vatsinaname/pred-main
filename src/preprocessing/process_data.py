import numpy as np
import pandas as pd
from scipy.fft import fft
import pywt
from sklearn.preprocessing import StandardScaler
import os

class SignalProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def apply_fft(self, signal, sampling_rate):
        """
        apply fft to signal - from signal processing course
        """
        n = len(signal)
        fft_result = fft(signal)
        freq = np.fft.fftfreq(n, 1/sampling_rate)
        return freq, np.abs(fft_result)
    
    def apply_wavelet_transform(self, signal, wavelet='db4', level=4):
        """
        do wavelet transform - kinda confusing but works?? 
        """
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        return coeffs
    
    def extract_time_domain_features(self, signal):
        """
        extract time features from signal - basic stats stuff
        """
        features = {
            'rms': np.sqrt(np.mean(np.square(signal))),  # root mean square
            'kurtosis': np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4),  # pointiness??
            'crest_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(np.square(signal)))  #peak to rms ratio
        }
        return features
    
    def extract_frequency_domain_features(self, signal, sampling_rate):
        """
        extract freq domain feats - FFT based
        """
        freq, fft_result = self.apply_fft(signal, sampling_rate)
        features = {
            'dominant_freq': freq[np.argmax(fft_result)],  # biggest frequency
            'spectral_entropy': -np.sum(fft_result * np.log2(fft_result + 1e-10))  # add eps to avoid log(0)
        }
        return features

class DataPreprocessor:
    def __init__(self):
        self.signal_processor = SignalProcessor()
        
    def load_ai4i_data(self, file_path):
        """
        load & preproc the ai4i data - mostly just normalization
        """
        df = pd.read_csv(file_path)
        
        # normalize cols (z-score)
        numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 
                         'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        df[numerical_cols] = self.signal_processor.scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def load_battery_data(self, file_path):
        """
        load battery data - this took 4ever to debug ughh
        """
        df = pd.read_csv(file_path)
        
        # normalize based on which cols we have (diff datasets!!)
        if 'Voltage' in df.columns:
            numerical_cols = ['Voltage', 'Current', 'Temperature']
        else:
            numerical_cols = ['c_vol', 'c_cur', 'c_surf_temp']
        
        df[numerical_cols] = self.signal_processor.scaler.fit_transform(df[numerical_cols])
        
        return df

if __name__ == "__main__":
    # test run
    preprocessor = DataPreprocessor()
    
    # try ai4i
    ai4i_data = preprocessor.load_ai4i_data("../../ai4i2020.csv")
    print("AI4I Dataset Shape:", ai4i_data.shape)
    
    # try battery - this one is huuuge btw
    battery_data = preprocessor.load_battery_data("../../Battery Dataset.csv")
    print("Battery Dataset Shape:", battery_data.shape) 