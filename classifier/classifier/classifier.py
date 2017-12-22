import numpy as np
from sklearn.metrics import log_loss

def lloss(yhat, y):
    n = yhat.shape[1]
    result = 0 
    for i in range(n):
        result += log_loss(y[:, i], yhat[:, i])

    return result/n
                
