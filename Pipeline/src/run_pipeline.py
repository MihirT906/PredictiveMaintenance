import pandas as pd
import numpy as np

from preprocessing import prepare_data, encode_labels
from smote_tomek_pipeline import create_pipeline, train_pipeline, evaluate_pipeline
from evaluation import get_metrics

#Retrieve data
df = pd.read_csv('../data/cleaned_data.csv', index_col=0) 

#Prepare data
unique_classes = np.unique(df['Failure_type'])

NUMERIC_FEATURES = ['Air_temperature', 'Process_temperature', 'Rotational_speed', 'Torque', 'Tool_wear']
CATEGORIC_FEATURES = ['Type']
X_train, X_test, y_train, y_test = prepare_data(df, NUMERIC_FEATURES + CATEGORIC_FEATURES, 'Failure_type')

#Label Encoding
y_train_encoded, label_encoder = encode_labels(y_train)
y_test_encoded, _ = encode_labels(y_test)

#Create and run pipeline
pipeline = create_pipeline(NUMERIC_FEATURES, CATEGORIC_FEATURES)
trained_pipeline = train_pipeline(pipeline, X_train, y_train_encoded)

metrics = evaluate_pipeline(trained_pipeline, X_test, y_test, label_encoder, unique_classes)
print(metrics)