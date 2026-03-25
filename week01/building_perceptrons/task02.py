from utils import create_dataset, initialize_weights
import numpy as np


def calculate_loss(w: float, dataset: list):
    return np.mean((dataset[:, 1] - w * dataset[:, 0]) ** 2)


def main():
    dataset = create_dataset(6)
    # old way of setting seed produces expeccted result
    np.random.seed(42)
    w = initialize_weights(0, 10)
    loss = calculate_loss(w - 0.001 * 2, dataset)
    print(f"MSE: {loss}")


if __name__ == "__main__":
    main()

# what happens to loss function when you pass
# `w + 0.001 * 2`, `w + 0.001`, `w - 0.001` and `w - 0.001 * 2`?
# 27.9896 (loss increases), 27.9576 (loss increases), 27.8936 (loss decreaase), 27.8616 (loss decrease)
# MSE is 0 when w is 2, currently it is 3.74
