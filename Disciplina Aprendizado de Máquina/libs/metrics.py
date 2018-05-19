import numpy as np

# MSE
def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)

# RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

# MAE
def mae(y_true, y_pred):
    return np.sum(np.absolute(y_true - y_pred)) / len(y_true)

# ACCURACY
def accuracy(y_true, y_pred):
    score = 0
    for i in range(len(y_true)):
        if (y_true[i] == y_pred[i]):
            score = score + 1
    return score/len(y_true)

# PRECISION
def precision(y_true, y_pred):
    tp = 0
    fp = 0
    
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                tp += 1
            else:
                fp +=1
            
    return tp / (tp + fp)

#RECALL
def recall(y_true, y_pred):
    tp =0
    fp = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                tp += 1
            else:
                fp +=1

    tn =0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 0:
            if y_pred[i] == 0:
                tn += 1
            else:
                fn +=1

    return (tp/(tp+fn))

# F1 MEASURE
def f1_measure(y_true, y_pred):
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    return 2 * (precision_ * recall_) / (precision_ + recall_)


