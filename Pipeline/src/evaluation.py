from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score

def get_metrics(y_true, y_pred, unique_classes):
    # Calculating F1 scores for each class
    f1_scores_per_class = f1_score(y_true, y_pred, average=None, labels=unique_classes)
    recall_scores_per_class = recall_score(y_true, y_pred, average=None, labels=unique_classes)
    precision_scores_per_class = precision_score(y_true, y_pred, average=None, labels=unique_classes)
    #class_f1_scores = dict(zip(unique_classes, f1_scores_per_class))
    class_recall_scores = dict(zip(unique_classes, recall_scores_per_class))
    class_precision_scores = dict(zip(unique_classes, precision_scores_per_class))
    dict_metrics = {
    'Accuracy': accuracy_score(y_true, y_pred),
    'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
    'Macro Recall': recall_score(y_true, y_pred, average='macro'), 
    'Macro Precision': precision_score(y_true, y_pred, average='macro'), 
    'Macro F1': f1_score(y_true, y_pred, average='macro'),
    # 'F1 Scores per Class': f1_scores_per_class,
    # 'Recall Scores per Class': class_recall_scores,
    # 'Precision Scores per Class': class_precision_scores
    }
    return dict_metrics