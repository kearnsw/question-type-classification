import torch
import numpy as np


def to_one_hot(matrix: torch.autograd.Variable, nb_classes: int) -> torch.autograd.Variable:
    """

    :param nb_classes:
    :param matrix: a matrix of class indices for  (batch_size, 1)
    :return:
    """
    class_matrix = matrix.data.numpy()
    oh_matrix = np.zeros((class_matrix.size, nb_classes))
    for i in range(class_matrix.size):
        oh_matrix[i][class_matrix[i]] = 1
        i += 1
    return torch.autograd.Variable(torch.from_numpy(oh_matrix)).type(torch.FloatTensor)


def to_k_hot(label_matrix: torch.autograd.Variable, nb_classes: int) -> torch.autograd.Variable:
    """
    Creates a matrix of size (nb_batches, nb_classes) where each row is a k-hot vector corresponding to the labels in
    the label matrix
    :param nb_classes:
    :param label_matrix: a matrix of class indices for  (batch_size, 1)
    :return:
    """
    class_matrix = label_matrix.data.numpy()

    oh_matrix = np.zeros((class_matrix.shape[0], nb_classes))
    for i in range(class_matrix.shape[0] - 1):
        for class_idx in class_matrix[i]:
            if class_idx != [-1]:
                oh_matrix[i][class_idx] = 1
        i += 1
    return torch.autograd.Variable(torch.from_numpy(oh_matrix).type(torch.FloatTensor))
