import torch
import torch.autograd as autograd
import torch.utils.data as torchdata
import numpy as np


class CharTrainer:
    """
    Generic class to train a model.
    Supports stopping and resuming the training, saves checkpoints,
    saves all training loss and validation accuracy data.
    """

    def __init__(self, model, optimizer, loss,
                 X, X2, y, train_pct, batch_size):
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

        # train/test split
        self.train_idx = int(y.shape[0]*train_pct)
        self.X_train = X[:self.train_idx, :].astype(int)
        self.X_test = X[self.train_idx:, :].astype(int)

        self.X2_train = X2[:self.train_idx, ...].astype(int)
        self.X2_test = X2[self.train_idx:, ...].astype(int)

        self.y_train = y[:self.train_idx, :].astype(int)
        self.y_test = y[self.train_idx:, :].astype(int)

        # creating torch data loaders
        train_dataset = torchdata.TensorDataset(torch.LongTensor(self.X_train),
                                                torch.LongTensor(self.y_train))
        test_dataset = torchdata.TensorDataset(torch.LongTensor(self.X_test),
                                               torch.LongTensor(self.y_test))
        train_dataset2 = torchdata.TensorDataset(
            torch.LongTensor(self.X2_train),
            torch.LongTensor(self.y_train))
        test_dataset2 = torchdata.TensorDataset(torch.LongTensor(self.X2_test),
                                                torch.LongTensor(self.y_test))

        self.train_loader = torchdata.DataLoader(
            train_dataset, batch_size=self.batch_size)
        self.train_loader2 = torchdata.DataLoader(
            train_dataset2, batch_size=self.batch_size)
        self.test_loader = torchdata.DataLoader(
            test_dataset, batch_size=self.batch_size)
        self.test_loader2 = torchdata.DataLoader(
            test_dataset2, batch_size=self.batch_size)

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
            epoch_loss = 0
            epoch_preds = []
            self.model.train()
            for (X, y), (X2, y2) in zip(self.train_loader, self.train_loader2):
                X, y = autograd.Variable(X).cuda(), autograd.Variable(y).cuda()
                X2 = autograd.Variable(X2).cuda()
                y = y.view(-1)
                self.optimizer.zero_grad()
                log_probs = self.model(X, X2)
                loss = self.loss(log_probs, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data[0]

                epoch_preds.append(
                    self.logprobs_to_predictions(log_probs).cpu())

            self.train_losses.append(epoch_loss)
            train_score = self.get_fscore(epoch_preds)
            self.train_scores.append(train_score)

            # validation
            train_preds = []
            self.model.eval()
            for (X, y), (X2, _) in zip(self.test_loader, self.test_loader2):
                X, y = autograd.Variable(X).cuda(), autograd.Variable(y).cuda()
                X2 = autograd.Variable(X2).cuda()
                y = y.view(-1)
                log_probs = self.model(X, X2)
                train_preds.append(
                    self.logprobs_to_predictions(log_probs).cpu())

            test_score = self.get_fscore(train_preds, train=False)
            self.test_scores.append(test_score)

            self.current_epoch += 1

            print('loss: {:04.2f}, score: {:04.2f}, val: {:04.2f}'.format(
                epoch_loss, train_score, test_score))

    def get_fscore(self, preds, train=True):
        """
        calculates a strange F score that we use as a proxy
        """
        tp, fp, fn = 0, 0, 0

        if train:
            dataloader = self.train_loader
        else:
            dataloader = self.test_loader

        for pred, (_, y) in zip(preds, dataloader):
            y = y.view(-1)
            tp += (pred == y)[y != 0].sum()
            fp += (pred != y)[y == 0].sum()
            fn += (pred != y)[y != 0].sum()

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 200 * precision * recall / (precision + recall)
        except:
            return 0.0

    def logprobs_to_predictions(self, logprobs):
        """
        returns a vector of predictions from the log probability tensor
        """
        return torch.max(logprobs, 1)[1].data

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

    def predict(self, X, X2):
        self.model.eval()
        y_empty = torch.zeros(X.shape[0])
        dataset = torchdata.TensorDataset(torch.LongTensor(X.astype(int)), y_empty)
        dataset2 = torchdata.TensorDataset(torch.LongTensor(X2.astype(int)), y_empty)
        loader = torchdata.DataLoader(dataset, batch_size=self.batch_size)
        loader2 = torchdata.DataLoader(dataset2, batch_size=self.batch_size)
        preds = []

        for (X, _), (X2, _) in zip(loader, loader2):
            X = autograd.Variable(X).cuda()
            X2 = autograd.Variable(X2).cuda()
            log_probs = self.model(X, X2)
            pred = self.logprobs_to_predictions(log_probs)
            pred = pred.view(-1, 50).cpu().numpy()
            preds.append(pred)

        return np.vstack(preds)
