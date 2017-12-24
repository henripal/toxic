import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.utils.data as torchdata
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import pdb


class CharTrainer:
    """
    Generic class to train a model.
    Supports stopping and resuming the training, saves checkpoints,
    saves all training loss and validation accuracy data.
    """

    def __init__(self, model, optimizer, loss,
                 X, X2, y, batch_size,
                 val_size=.2, stratify=None, X_test=None):
        """
        model: model to train
        optimizer: chosen optimizer
        X, y: data (not split), expects numpy arrays
        """
        assert X.shape[0] == y.shape[0]

        self.model = model
        if self.model.gpu:
            model = model.cuda()
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size

        # train/test split
        self.X_train, self.X_val, self.X2_train, self.X2_val, self.y_train, self.y_val = train_test_split(
                X, X2, y, stratify=stratify, test_size=val_size)

        # creating torch data loaders
        train_dataset = torchdata.TensorDataset(torch.LongTensor(self.X_train),
                                                torch.LongTensor(self.y_train))
        val_dataset = torchdata.TensorDataset(torch.LongTensor(self.X_val),
                                               torch.LongTensor(self.y_val))
        train_dataset2 = torchdata.TensorDataset(
            torch.LongTensor(self.X2_train),
            torch.LongTensor(self.y_train))
        val_dataset2 = torchdata.TensorDataset(torch.LongTensor(self.X2_val),
                                                torch.LongTensor(self.y_val))

        self.train_loader = torchdata.DataLoader(
            train_dataset, batch_size=self.batch_size)
        self.train_loader2 = torchdata.DataLoader(
            train_dataset2, batch_size=self.batch_size)
        self.val_loader = torchdata.DataLoader(
            val_dataset, batch_size=self.batch_size)
        self.val_loader2 = torchdata.DataLoader(
            val_dataset2, batch_size=self.batch_size)

        # history data
        self.train_losses = []
        self.train_scores = []
        self.test_scores = []

        self.current_epoch = 0
        self.resume = False

    def train(self, epochs=10, save_freq=10):
        """
        starts or resumes training.
        saves everything every save_freq minutes
        """

        self.resume = True

        for epoch in range(epochs):
            print('Epoch: {:<4}'.format(self.current_epoch), end=' ')
            # training:
            epoch_losses = []
            self.model.train()
            for (X, y), (X2, y2) in zip(self.train_loader, self.train_loader2):
                if self.model.gpu:
                    X, y = autograd.Variable(X).cuda(), autograd.Variable(y).cuda()
                    X2 = autograd.Variable(X2).cuda()
                else:
                    X, y = autograd.Variable(X), autograd.Variable(y)
                    X2 = autograd.Variable(X2)

                self.optimizer.zero_grad()
                log_probs = self.model(X, X2)
                loss = self.loss(log_probs, y.float())
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.data[0])

            epoch_loss = np.mean(epoch_losses)
            self.train_losses.append(epoch_loss)

            # validation
            val_losses = []
            self.model.eval()
            for (X, y), (X2, _) in zip(self.val_loader, self.val_loader2):
                if self.model.gpu:
                    X, y = autograd.Variable(X).cuda(), autograd.Variable(y).cuda()
                    X2 = autograd.Variable(X2).cuda()
                else:
                    X, y = autograd.Variable(X), autograd.Variable(y)
                    X2 = autograd.Variable(X2)

                log_probs = self.model(X, X2)
                loss = self.loss(log_probs, y.float())
                val_losses.append(loss.cpu().data[0])

            val_loss = np.mean(val_losses)

            self.current_epoch += 1

            print('loss: {:06.4f},  val: {:06.4f}'.format(
                epoch_loss, val_loss))


    def test_predict(self):
        self.model.eval()
        preds = []
        for (X, y), (X2, _) in zip(self.test_loader, self.test_loader2):
            X, y = autograd.Variable(X).cuda(), autograd.Variable(y).cuda()
            X2 = autograd.Variable(X2).cuda()
            y = y.view(-1)
            log_probs = self.model(X, X2)
            pred = self.logprobs_to_predictions(log_probs)
            pred = pred.view(-1, 50).cpu().numpy()
            preds.append(pred)

        return np.vstack(preds)

