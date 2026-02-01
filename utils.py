import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from transformers import EvalPrediction

def load_data(file_path, remove_placeholder_tokens=False):
    df = pd.read_csv(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['target'],
        test_size=0.2,
        stratify=df['target'],
        random_state=42
    )

    # Remove placeholder tokens
    if remove_placeholder_tokens:
        X_train = X_train.replace('<LINK>', '').replace('<MENTION>', '')
        X_test = X_test.replace('<LINK>', '').replace('<MENTION>', '')

    print("train distribution:")
    print(f"true:  {(y_train == 1).sum()/len(y_train)*100}%")
    print(f"false: {(y_train == 0).sum()/len(y_train)*100}%")
    print("\ntest distribution:")
    print(f"true:  {(y_test == 1).sum()/len(y_test)*100}%")
    print(f"false: {(y_test == 0).sum()/len(y_test)*100}%")

    return df, X_train, X_test, y_train, y_test

def compute_metrics(eval_pred: EvalPrediction) -> dict:
    # predictions = [[percentage_0_class_elem_1, percentage_1_class_elem_1], ...]
    # labels = [true_label_elem_1, true_label_elem_2, ...]
    predictions, labels = eval_pred
    
    # get the position of the bigger percentage (class 0 or 1); axis=1 -> per element
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary' # treat the labels together as a binary class, not separate
    )
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }