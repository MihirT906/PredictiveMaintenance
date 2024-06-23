from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

def create_smote_tomek_pipeline(preprocessor, model):
    sampling_strategy = {
        0: 7735,   # HDF 
        1: 7735,   # NF 
        2: 7735,   # OSF
        3: 7735,   # PWF
        4: 100000    # TWF
    }
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote_tomek', SMOTETomek(sampling_strategy=sampling_strategy, random_state=2023)),  
        ('model', model)
    ])

    return pipeline