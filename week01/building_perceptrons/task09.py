import numpy as np
from utils import (
    initialize_weights,
    create_dataset_and,
    create_dataset_nand,
    calculate_appr_der_logistic,
    calculate_loss_logistic,
    sigmoid,
)
import matplotlib.pyplot as plt
 
# AND w1, w2, w = (w1, w2, b1), x = (x1, x2)
# y = sig(wx + b)
# NAND w = (-w1, -w2, -b1)

EPSILON = 1e-6

def main():
    lr = 0.05
    # AND
    w_and = initialize_weights(-1, 1, 3)  # + bias at the end
    dataset_and = create_dataset_and()

    for _ in range(0, 100001):
        grad = calculate_appr_der_logistic(w_and, dataset_and, EPSILON)
        w_and -= grad * lr

    # pure nand
    w_nand = initialize_weights(-1, 1, 3)  # + bias at the end
    dataset_nand = create_dataset_nand()

    for _ in range(0, 100001):
        grad = calculate_appr_der_logistic(w_nand, dataset_nand, EPSILON)
        w_nand -= grad * lr


    # trained
    print("AND:", sigmoid(dataset_and[:, :-1] @ w_and[:-1] + w_and[-1]))
    print("NAND using AND model:", sigmoid(dataset_and[:, :-1] @ (-w_and[:-1]) - w_and[-1]))
    print("NAND:", sigmoid(dataset_nand[:, :-1] @ w_nand[:-1] + w_nand[-1]))


if __name__ == "__main__":
    main()
