import numpy as np
from sklearn.metrics import log_loss
import pandas as pd
import torch.utils.data as torchdata
import torch
import torch.autograd as autograd

def lloss(yhat, y):
    n = yhat.shape[1]
    result = 0
    for i in range(n):
        result += log_loss(y[:, i], yhat[:, i])

    return result/n

def make_submission(lang, model, sub, filename, datapath, comment_types):
    X_test = lang.encoded_test_sentences
    y_empty = torch.zeros(X_test.shape[0])
    dataset = torchdata.TensorDataset(torch.LongTensor(X_test.astype(int)), y_empty)
    loader = torchdata.DataLoader(dataset, batch_size = 256)
    preds = []
    model.eval()

    print('predicting...')
    for X, _ in loader:
        X = autograd.Variable(X).cuda()
        log_probs = model(X).cpu().data.numpy()
        preds.append(log_probs)
    final = np.vstack(preds)

    print('saving file...')
    sub.loc[:, comment_types] = final
    sub.to_csv(datapath + filename, index = False)
    print('done')
    return final

