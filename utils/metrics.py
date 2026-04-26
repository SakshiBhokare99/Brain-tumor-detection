from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(y_true,y_pred):
    p = precision_score(y_true,y_pred)
    r = recall_score(y_true,y_pred)
    f = f1_score(y_true,y_pred)

    print("Precision:",p)
    print("Recall:",r)
    print("F1 Score:",f)