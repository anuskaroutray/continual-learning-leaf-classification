import os
import argparse

from utils import set_seed, Trainer
from data import DeepHerbDataset

from torch.utils.data import DataLoader

from src import BaselineModel
import torch
import torch.nn as nn

from utils import class_incremental_dataset, SoftTarget
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR

import copy

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default = "resnet50", type = str)
    ap.add_argument("--model-save-name", default = "Baseline_ResNet", type = str)
    ap.add_argument("--data-dir", default = "/home/sohan/scratch/deepherb/Medicinal Leaf Dataset/Segmented Medicinal Leaf Images", type = str)
    ap.add_argument("--num-classes", default = 30, type = int)
    ap.add_argument("--csv-name", default = "leaf-data.csv", type = str)
    ap.add_argument("--seed", default = 0, type = int)
    ap.add_argument("--batch-size", default = 128, type = int)
    ap.add_argument("--pretrained", default = True, type = bool)
    ap.add_argument("--lr", default = 1e-4, type = float)
    ap.add_argument("--weight-decay", default = 1e-4, type = float)
    ap.add_argument("--loss-function", default = "cross_entropy_loss", type = str)
    ap.add_argument("--num-workers", default = 1, type = int)
    ap.add_argument("--scheduler", default = "CosineAnnealingScheduler", type = str)
    ap.add_argument("--epochs", default = 5, type = int)


    ap.add_argument("--incremental-learning", default = True, type = bool)
    ap.add_argument('--num-base-classes', default = 15, type = int)
    ap.add_argument('--increment', default = 5, type = int)
    ap.add_argument('--use-kldiv', default = True, type = bool)
    ap.add_argument('--use-teacher', default = True, type = bool)

    # data_dir, csv_name = None, num_classes = None
    args = ap.parse_args()

    if args.incremental_learning:
        assert (args.use_kldiv and args.use_teacher) or (not args.use_kldiv and not args.use_teacher), "Both KL Divergence and Teacher should be of same boolean nature"


    set_seed(args.seed)

    train_dataset = DeepHerbDataset(args.data_dir, num_classes = args.num_classes, mode = "train")
    test_dataset = DeepHerbDataset(args.data_dir, num_classes = args.num_classes, mode = "test")

    model = BaselineModel(args.model_name, pretrained = args.pretrained, num_classes = args.num_classes)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr = float(args.lr), weight_decay = float(args.weight_decay))

    if not args.incremental_learning:

        train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

        criterion = None
        if args.loss_function == "cross_entropy_loss":
            criterion = nn.CrossEntropyLoss()

        trainer = Trainer(args, model, optimizer, criterion, device)
        trainer.train(train_dataloader)
        train_metrics = trainer.compute_metrics(train_dataloader)
        test_metrics = trainer.compute_metrics(test_dataloader)
        print(f"\nTrain metrics: {train_metrics}")
        print(f"\nTest metrics: {test_metrics}\n")

    elif args.incremental_learning:

        class_order = np.random.permutation(args.num_classes)

        train_dataset, train_class_list = class_incremental_dataset(train_dataset, class_order, is_train = True, 
                                                                    num_base_classes = args.num_base_classes, increment = args.increment)
        test_dataset, test_class_list = class_incremental_dataset(test_dataset, class_order, is_train = False, 
                                                                    num_base_classes = args.num_base_classes, increment = args.increment)

        train_dataloader = [DataLoader(data, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers) for data in train_dataset]
        test_dataloader = [DataLoader(data, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers) for data in test_dataset]

        criterion = None
        if args.loss_function == "cross_entropy_loss":
            criterion = nn.CrossEntropyLoss()
        
        criterion_kldiv = None
        if args.use_kldiv:
            criterion_kldiv = SoftTarget(T = 2)

        teacher_model = None
        trainer = Trainer(args, model, optimizer, criterion, device, criterion_kldiv = criterion_kldiv)

        for i, (train_loader, test_loader) in enumerate(zip(train_dataloader, test_dataloader)):
            print(f"Training Model on Task {i+1}")
            print(f"Classes on which the model is being trained: {train_class_list[i]}")
            print(f"Classes on which the model is being evaluated: {test_class_list[i]}")

            trainer.train(train_loader, teacher_model = teacher_model, known_classes = test_class_list)

            train_metrics = trainer.compute_metrics(train_loader)
            test_metrics = trainer.compute_metrics(test_loader)
            print(f"\nTrain metrics for task {i+1}: {train_metrics}")
            print(f"\nTest metrics for task {i+1}: {test_metrics}\n")

            if args.use_teacher:
                teacher_model = copy.deepcopy(trainer.model)
                for name, param in teacher_model.named_parameters():
                    param.requires_grad = False

            trainer.args.epochs = 10
            trainer.scheduler = CosineAnnealingLR(trainer.optimizer, trainer.args.epochs)

    torch.save(model, f"./checkpoints/{args.model_save_name}.pth")