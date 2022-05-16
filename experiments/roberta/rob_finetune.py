import transformers
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    metrics = precision_recall_fscore_support(labels, preds)
    met = {'accuracy': accuracy_score(labels, preds),
           'f1': metrics[2][1],
           'precision': metrics[0][1],
           'recall': metrics[1][1]}
    return met

def model_init():
    """Model initialization"""
    model = transformers.RobertaForSequenceClassification.from_pretrained("roberta-base")
    return model
