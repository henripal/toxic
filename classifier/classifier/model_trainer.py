import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.utils.data as torchdata
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split


class Trainer:
    """
    Generic class to train a model.
    Supports stopping and resuming the training, saves checkpoints,
    saves all training loss and validation accuracy data.
    """

    def __init__(self, model, optimizer, loss,
                 X, y, batch_size,
                 val_size=0.2, stratify=None,
                 gpu=False,
                 X_test=None):
        """
        model: model to train
        optimizer: chosen optimizer
        X, y: data (not split), expects numpy arrays
        """
        assert X.shape[0] == y.shape[0]

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size

        self.gpu = gpu

        # train/test split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X,
                y,
                stratify=stratify,
                test_size=val_size)

        # creating torch data loaders
        train_dataset = torchdata.TensorDataset(torch.LongTensor(self.X_train),
                                                torch.LongTensor(self.y_train))
        val_dataset = torchdata.TensorDataset(torch.LongTensor(self.X_val),
                                                torch.LongTensor(self.y_val))

        self.train_loader = torchdata.DataLoader(train_dataset, batch_size = self.batch_size)
        self.val_loader = torchdata.DataLoader(val_dataset, batch_size = self.batch_size)

        # history data
        self.train_losses = []
        self.train_scores = []
        self.val_scores = []

        self.current_epoch = 0
        self.resume = False

    def train(self, epochs=10, save_freq=10):
        """
        starts or resumes training.
        """

        self.resume = True

        for epoch in range(epochs):
            print('Epoch: {:<4}'.format(self.current_epoch), end=' ')
            # training:
            epoch_losses = []
            self.model.train()
            for X, y in self.train_loader:
                if self.gpu:
                    X, y = autograd.Variable(X).cuda(), autograd.Variable(y).cuda()
                else:
                    X, y = autograd.Variable(X), autograd.Variable(y)
                self.optimizer.zero_grad()
                log_probs = self.model(X)
                loss = self.loss(log_probs, y.float())
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.data[0])


            epoch_loss = np.mean(epoch_losses)
            self.train_losses.append(epoch_loss)

            # validation
            val_losses = []
            self.model.eval()
            for X, y in self.val_loader:
                if self.gpu:
                    X, y = autograd.Variable(X).cuda(), autograd.Variable(y).cuda()
                else:
                    X, y = autograd.Variable(X), autograd.Variable(y)
                log_probs = self.model(X)
                loss = self.loss(log_probs, y.float())
                val_losses.append(loss.cpu().data[0])

            val_loss = np.mean(val_losses)

            self.current_epoch += 1

            print('loss: {:06.4f}, val: {:06.4f}'.format(epoch_loss,
                                                                 val_loss))



    def test_predict(self):
        self.model.eval()
        preds = []
        for X, y in self.test_loader:
            X, y = autograd.Variable(X).cuda(), autograd.Variable(y).cuda()
            y = y.view(-1)
            log_probs = self.model(X)
            pred = pred.view(-1, 50).cpu().numpy()
            preds.append(pred)

        return np.vstack(preds)







