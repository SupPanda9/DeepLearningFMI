import numpy as np
from utils import initialize_weights


def create_dataset_and():
    return np.array([(i, j, i * j) for i in range(0, 2) for j in range(0, 2)])


def create_dataset_or():
    return np.array([(i, j, i | j) for i in range(0, 2) for j in range(0, 2)])


def calculate_loss_multiple_inputs(w: float, dataset: np.ndarray):
    label = dataset[:, -1]
    prediction = dataset[:, :-1] @ w
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


# AND w1, w2, w = (w1, w2), x = (x1, x2)
# y = wx

# OR w1, w2, w = (w1, w2), x = (x1, x2)

EPSILON = 1e-6


def main():
    lr = 0.001
    # AND
    w_and = initialize_weights(0, 10, 2)
    dataset_and = create_dataset_and()

    for i in range(0, 100000):
        grad = calculate_appr_derivative_multiple_inputs(w_and, dataset_and, EPSILON)
        w_and -= grad * lr
        print(
            f"[{i}] weights: ",
            w_and,
            " loss:",
            calculate_loss_multiple_inputs(w_and, dataset_and),
        )

    # OR
    w_or = initialize_weights(0, 10, 2)
    dataset_or = create_dataset_or()

    for i in range(0, 100000):
        grad = calculate_appr_derivative_multiple_inputs(w_or, dataset_or, EPSILON)
        w_or -= grad * lr
        print(
            f"[{i}] weights: ",
            w_or,
            " loss:",
            calculate_loss_multiple_inputs(w_or, dataset_or),
        )

    # trained
    print("AND:", w_and @ dataset_and[:, :-1].transpose())
    print("OR:", w_or @ dataset_or[:, :-1].transpose())
    # AND: [0.       0.333333 0.333333 0.666666]
    # OR: [0.         0.66666633 0.66666633 1.33333267]
    # so not confident at all, and loss is not almost 0


if __name__ == "__main__":
    main()
