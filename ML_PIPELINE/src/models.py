from xgboost import XGBClassifier

def get_model(model_name):
    if model_name == 'xgboost':
        return XGBClassifier(random_state=2023)
    # Add more models here as needed
    else:
        raise ValueError("Model not recognized.")
