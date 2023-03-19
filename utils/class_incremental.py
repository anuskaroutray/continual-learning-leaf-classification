import numpy as np
import torch
import torch.nn as nn
import copy


def class_incremental_dataset(dataset, class_order, is_train = True, num_base_classes = 15, increment = 5):

    """
        Function to build the class incremental dataset
        Returns:
            List of datasets, [base dataset, dataset with class incremented by incremental step ....]
    """

    num_classes = len(class_order)

    dataset_list = []
    class_list = []
    for i in range(num_base_classes, num_classes+1, increment):
        
        dataset_copy = copy.deepcopy(dataset)
        if is_train:
            if i == num_base_classes:
                dataset_copy.df = dataset_copy.df[dataset_copy.df["label"].isin(class_order[:i])]
                classes = class_order[:i]
            else:
                dataset_copy.df = dataset_copy.df[dataset_copy.df["label"].isin(class_order[i-increment:i])]
                classes = class_order[i-increment:i]

        else:
            dataset_copy.df = dataset_copy.df[dataset_copy.df["label"].isin(class_order[:i])]
            classes = class_order[:i]

        dataset_copy.df.reset_index(inplace = True)

        dataset_list.append(dataset_copy)
        class_list.append(classes)

    return dataset_list, class_list