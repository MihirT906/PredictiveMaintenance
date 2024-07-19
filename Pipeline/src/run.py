import pandas as pd
import numpy as np

from preprocessing import prepare_data, encode_labels
from pipeline import baseline_pipeline, smote_tomek_pipeline, smote_enn_pipeline
from evaluation import get_metrics
import mlflow
import mlflow.sklearn


def log_run(model, name, NUMERIC_FEATURES, CATEGORIC_FEATURES, sampling_strategy, X_train, y_train_encoded, label_encoder, unique_classes):
    mlflow.start_run(run_name = f"{name}_run")
    pipeline = model.create_pipeline(NUMERIC_FEATURES, CATEGORIC_FEATURES, sampling_strategy)
    trained_pipeline = model.train_pipeline(pipeline, X_train, y_train_encoded)

    metrics = model.evaluate_pipeline(trained_pipeline, X_test, y_test, label_encoder, unique_classes)
    mlflow.log_param("NUMERIC_FEATURES", NUMERIC_FEATURES)
    mlflow.log_param("CATEGORIC_FEATURES", CATEGORIC_FEATURES)
    mlflow.sklearn.log_model(trained_pipeline, f"{name} Pipeline")
    mlflow.log_metrics(metrics)
    mlflow.end_run()

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

mlflow.set_tracking_uri(uri="")
exp = mlflow.set_experiment(experiment_name="Predictive Maintenance Experiment")

#Create and run pipeline - base pipeline
model = baseline_pipeline()
log_run(model, "baseline_XGB", NUMERIC_FEATURES, CATEGORIC_FEATURES, None, X_train, y_train_encoded, label_encoder, unique_classes)


#Create and run pipeline - smote tomek pipeline (custom sampling strategy)
model = smote_tomek_pipeline()
log_run(model, "custom_smote_tomek_XGB", NUMERIC_FEATURES, CATEGORIC_FEATURES, "custom", X_train, y_train_encoded, label_encoder, unique_classes)
log_run(model, "auto_smote_tomek_XGB", NUMERIC_FEATURES, CATEGORIC_FEATURES, "auto", X_train, y_train_encoded, label_encoder, unique_classes)


#Create and run pipeline - smote enn pipeline (custom sampling strategy)
model = smote_enn_pipeline()
log_run(model, "custom_smote_enn_XGB", NUMERIC_FEATURES, CATEGORIC_FEATURES, "custom", X_train, y_train_encoded, label_encoder, unique_classes)
log_run(model, "auto_smote_enn_XGB", NUMERIC_FEATURES, CATEGORIC_FEATURES, "auto", X_train, y_train_encoded, label_encoder, unique_classes)


# mlflow.start_run(run_name = "baseline_XGB_run")
# model = baseline_pipeline()
# pipeline = model.create_pipeline(NUMERIC_FEATURES, CATEGORIC_FEATURES)
# trained_pipeline = model.train_pipeline(pipeline, X_train, y_train_encoded)

# metrics = model.evaluate_pipeline(trained_pipeline, X_test, y_test, label_encoder, unique_classes)
# mlflow.log_param("NUMERIC_FEATURES", NUMERIC_FEATURES)
# mlflow.log_param("CATEGORIC_FEATURES", CATEGORIC_FEATURES)
# mlflow.sklearn.log_model(trained_pipeline, "Baseline XGB Pipeline")
# mlflow.log_metrics(metrics)
# mlflow.end_run()