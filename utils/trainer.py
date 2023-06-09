import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import matplotlib.pyplot as plt

class Trainer():

    def __init__(self, args, model, optimizer, criterion, device, criterion_kldiv = None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.criterion_kldiv = criterion_kldiv

        self.scheduler = None

        if args.scheduler == "CosineAnnealingScheduler":
            self.scheduler = CosineAnnealingLR(self.optimizer, self.args.epochs)

        self.model.to(self.device)
        self.train_loss_list = []
        self.eval_loss_list = []

    def _train_step(self, dataloader, epoch, teacher_model = None, known_classes = None):
        
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        for batch_idx, batch in enumerate(tepoch):
            tepoch.set_description(f"INFO: Epoch {epoch + 1}")

            image, label = batch
            out = self.model(image.to(self.device))

            loss = self.criterion(out, label.to(self.device))
            if self.args.incremental_learning and teacher_model is not None:
                teacher_out = teacher_model(image.to(self.device))
                loss += 0.5 * self.criterion_kldiv(out[:, known_classes], teacher_out[:, known_classes])

            loss.backward()

            total_loss += loss.item()
            tepoch.set_postfix(loss = total_loss / (batch_idx + 1))
            
            self.optimizer.step()
            self.optimizer.zero_grad()

        return (total_loss / (batch_idx + 1))
    
    def train(self, train_dataloader, val_dataloader = None, teacher_model = None, known_classes = None):
        self.model.train()
        for epoch in range(self.args.epochs):
            self.train_loss_list.append(self._train_step(train_dataloader, epoch, teacher_model = teacher_model))

            if self.scheduler is not None:
                self.scheduler.step()
                
                self.model.train()

            self.eval_loss_list.append(self.evaluate(val_dataloader))

    def evaluate(self, dataloader):
        self.model.eval()
        
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        tepoch.set_description("INFO: Validation Step")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tepoch):
                
                image, label = batch
                out = self.model(image.to(self.device))

                loss = self.criterion(out, label.to(self.device))

                total_loss += loss.item()
                tepoch.set_postfix(loss = total_loss / (batch_idx+1))

        return (total_loss / (batch_idx + 1))

    def compute_metrics(self, dataloader):

        batch_accuracy = []
        batch_precision = []
        batch_recall = []
        batch_f1_score = []
        self.model.eval()
        
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        tepoch.set_description("Computing metrics")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tepoch):
                    
                image, label = batch
                out = self.model(image.to(self.device))
                out = torch.argmax(out, dim = 1)

                batch_accuracy.append(self._accuracy(out, label))
                batch_precision.append(self._precision(out, label))
                batch_recall.append(self._recall(out, label))
                batch_f1_score.append(self._f1_score(out, label))

        results = {
            "accuracy": np.mean(batch_accuracy),
            "precision": np.mean(batch_precision),
            "recall": np.mean(batch_recall),
            "f1_score": np.mean(batch_f1_score),
        }

        return results

    def _accuracy(self, outputs, labels):
        # NOTE: Weighted accuracy can be computed later
        # TODO: Implement class wise accuracy
        outputs = outputs.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        # print(outputs, labels)
        return accuracy_score(outputs, labels)

    def _precision(self, outputs, labels):
        # NOTE: Weighted precision can be computed later
        # TODO: Implement class wise precision
        outputs = outputs.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        return precision_score(outputs, labels, average='micro')

    def _recall(self, outputs, labels):
        # NOTE: Weighted recall can be computed later
        # TODO: Implement class wise recall
        outputs = outputs.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        return recall_score(outputs, labels, average='micro')

    def _f1_score(self, outputs, labels):
        # NOTE: Weighted f1_score can be computed later
        # TODO: Implement class wise f1_score
        outputs = outputs.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        return f1_score(outputs, labels, average='micro')

    def plot_loss(self, task = None):

        plt.figure(figsize = (6, 5))
        epochs = [i+1 for i in range(self.args.epochs)]
        plt.plot(epochs, self.train_loss_list, label = "Train")
        plt.plot(epochs, self.eval_loss_list, label = "Val")
        plt.title(f"Variation of loss over number of epochs")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")
        plt.legend()
        save_name = self.args.model_save_name
        if task is not None:
            save_name += f"_task{task}"
        plt.savefig(f"plots/{save_name}_loss.png")
