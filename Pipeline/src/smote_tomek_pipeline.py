from sklearn.pipeline import Pipeline

from preprocessing import create_preprocessor
from evaluation import get_metrics
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

def create_pipeline(numeric_features, cat_features):
    preprocessor = create_preprocessor(numeric_features, cat_features)
    sampling_strategy = {
        0: 7735,   # HDF 
        1: 7735,   # NF 
        2: 7735,   # OSF
        3: 7735,   # PWF
        4: 100000    # TWF
    }
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote_tomek', SMOTETomek(sampling_strategy=sampling_strategy, random_state=2023)),  # Adjust sampling_strategy as needed
        ('model', XGBClassifier(random_state=2023, objective='multi:softmax'))
    ])
    return pipeline

def train_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_pipeline(pipeline, X_test, y_test, label_encoder, unique_classes):
    y_pred_encoded = pipeline.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    metrics = get_metrics(y_test, y_pred, unique_classes)
    return metrics
