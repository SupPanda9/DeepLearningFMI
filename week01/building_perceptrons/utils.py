import numpy as np


def create_dataset(n: int):
    return np.array([(i, i * 2) for i in range(0, n)])


def initialize_weights(x: int, y: int, n=1):
    return np.random.uniform(x, y, n)


def calculate_loss(w: float, dataset: np.ndarray):
    return np.mean((dataset[:, 1] - w * dataset[:, 0]) ** 2)


def calculate_appr_derivative(w: float, dataset: np.ndarray, eps: float):
    return (calculate_loss(w + eps, dataset) - calculate_loss(w, dataset)) / eps


def create_dataset_and():
    return np.array([(i, j, i * j) for i in range(0, 2) for j in range(0, 2)])


def create_dataset_or():
    return np.array([(i, j, i | j) for i in range(0, 2) for j in range(0, 2)])


def create_dataset_nand():
    return np.array([(i, j, not (i * j)) for i in range(0, 2) for j in range(0, 2)])

def create_dataset_xor():
    return np.array([(i, j, i == j) for i in range(0, 2) for j in range(0, 2)])


def calculate_loss_multiple_inputs(w: float, dataset: np.ndarray):
    weights = w[:-1]
    bias = w[-1]
    label = dataset[:, -1]
    prediction = dataset[:, :-1] @ weights + bias
    return np.mean((label - prediction) ** 2)


def calculate_appr_derivative_multiple_inputs(
    w: float, dataset: np.ndarray, eps: float
):
    grad = np.zeros_like(w)

    for i in range(0, len(w)):
        w_eps = w.copy()
        w_eps[i] += eps

        grad[i] = (
            calculate_loss_multiple_inputs(w_eps, dataset)
            - calculate_loss_multiple_inputs(w, dataset)
        ) / eps

    return grad


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calculate_loss_logistic(w: float, dataset: np.ndarray):
    weights = w[:-1]
    bias = w[-1]
    label = dataset[:, -1]

    prediction = sigmoid(dataset[:, :-1] @ weights + bias)
    return np.mean((label - prediction) ** 2)

def calculate_appr_der_logistic(w: float, dataset: np.ndarray, eps: float):
    grad = np.zeros_like(w)

    for i in range(0, len(w)):
        w_eps = w.copy()
        w_eps[i] += eps

        grad[i] = (
            calculate_loss_logistic(w_eps, dataset)
            - calculate_loss_logistic(w, dataset)
        ) / eps

    return grad