import numpy as np
from utils import (
    initialize_weights,
    create_dataset_and,
    create_dataset_or,
    calculate_appr_der_logistic,
    calculate_loss_logistic,
    sigmoid,
)
import matplotlib.pyplot as plt

# AND w1, w2, w = (w1, w2, b1), x = (x1, x2)
# y = sig(wx + b)
# OR w1, w2, w = (w1, w2, b1), x = (x1, x2)
# y = sig(wx + b)
# 3 params each

EPSILON = 1e-6


def main():
    lr = 0.05
    # AND
    w_and = initialize_weights(-1, 1, 3)  # + bias at the end
    dataset_and = create_dataset_and()
    loss_and = list()

    for i in range(0, 100001):
        grad = calculate_appr_der_logistic(w_and, dataset_and, EPSILON)
        w_and -= grad * lr
        loss_and.append(calculate_loss_logistic(w_and, dataset_and))
        print(
            f"[{i}] weights: ",
            w_and,
            " loss:",
            loss_and[i],
        )

    # OR
    w_or = initialize_weights(-1, 1, 3)  # and bias
    dataset_or = create_dataset_or()
    loss_or = list()

    for i in range(0, 100001):
        grad = calculate_appr_der_logistic(w_or, dataset_or, EPSILON)
        w_or -= grad * lr
        loss_or.append(calculate_loss_logistic(w_or, dataset_or))
        print(
            f"[{i}] weights: ",
            w_or,
            " loss:",
            loss_or[i],
        )

    # trained
    print("AND:", sigmoid(dataset_and[:, :-1] @ w_and[:-1] + w_and[-1]))
    print("OR:", sigmoid(dataset_or[:, :-1] @ w_or[:-1] + w_or[-1]))

    plt.plot(np.arange(0, 100001, 1), loss_and)
    plt.title("Loss for AND function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(np.arange(0, 100001, 1), loss_or)
    plt.title("Loss for OR function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Results before sigmoid
    # AND: [-0.2500005  0.2499995  0.2499995  0.7499995]
    # OR: [0.2499995 0.7499995 0.7499995 1.2499995]

    # Results after
    # AND: [5.17766999e-05 3.39300800e-02 3.39300800e-02 9.59714722e-01]
    # OR: [0.03357254 0.97892459 0.97892456 0.9999839 ]

    # Looking at the results, the sigmoid introduces the
    # needed non-linearity to successfuly create the gates


if __name__ == "__main__":
    main()
