import numpy as np
from sklearn.metrics import log_loss
import pandas as pd
import torch.utils.data as torchdata
import torch
import torch.autograd as autograd
import pdb

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

def make_submission_char(lang, model, sub, filename, datapath, comment_types):
    X_test = lang.encoded_test_sentences
    X2_test = lang.encoded_test_chars
    y_empty = torch.zeros(X_test.shape[0])
    dataset = torchdata.TensorDataset(torch.LongTensor(X_test.astype(int)), y_empty)
    dataset2 = torchdata.TensorDataset(torch.LongTensor(X2_test.astype(int)), y_empty)
    loader = torchdata.DataLoader(dataset, batch_size = 256)
    loader2 = torchdata.DataLoader(dataset2, batch_size = 256)
    preds = []
    model.eval()

    print('predicting...')
    for (X, y), (X2, _) in zip(loader, loader2):
        Xc = autograd.Variable(X, volatile=True).cuda()
        X2c = autograd.Variable(X2, volatile=True).cuda()
        log_probs = model(Xc, X2c).cpu().data.numpy()
        preds.append(log_probs)
    final = np.vstack(preds)

    print('saving file...')
    sub.loc[:, comment_types] = final
    sub.to_csv(datapath + filename, index = False)
    print('done')
    return final

