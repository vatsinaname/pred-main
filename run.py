import os
import sys
import subprocess
import argparse

def check_environment():
    """check deps - make sure everything is installed"""
    try:
        import numpy
        import pandas
        import sklearn
        import pywt
        import flask
        print("✅ All required packages are installed.")
        return True
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def train_models():
    """train the ML models - takes a while!!"""
    print("Training models...")
    
    # add src to path so imports work right
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
    
    # make sure we have models dir
    os.makedirs('models', exist_ok=True)
    
    try:
        # run the train script - does all the hard work
        train_script = os.path.join('src', 'models', 'train.py')
        subprocess.run([sys.executable, train_script], check=True)
        print("✅ Models trained successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error training models.")
        return False

def start_webapp():
    """start flask app - needs internet connection"""
    print("Starting web dashboard...")
    try:
        # run flask
        app_script = os.path.join('web_dashboard', 'app.py')
        subprocess.run([sys.executable, app_script], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Error starting web dashboard.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive Maintenance Project Runner")
    parser.add_argument("--train", action="store_true", help="Train the models")
    parser.add_argument("--start", action="store_true", help="Start the web dashboard")
    
    args = parser.parse_args()
    
    # show help if no args given
    if not args.train and not args.start:
        print("Predictive Maintenance of Machine Components using ML")
        print("\nUsage options:")
        print("  1. Train models:   python run.py --train")
        print("  2. Start webapp:   python run.py --start")
        print("  3. Complete setup: python run.py --train --start")
        sys.exit(0)
    
    # check if all deps r installed
    if not check_environment():
        sys.exit(1)
    
    # train if --train arg given
    if args.train:
        if not train_models():
            sys.exit(1)
    
    # start app if --start arg given
    if args.start:
        start_webapp() 