import numpy as np
from utils import (
    initialize_weights,
    create_dataset_and,
    create_dataset_or,
    calculate_appr_derivative_multiple_inputs,
    calculate_loss_multiple_inputs,
)

# AND w1, w2, w = (w1, w2, b1), x = (x1, x2)
# y = wx + b
# OR w1, w2, w = (w1, w2, b1), x = (x1, x2)
# y = wx + b
# 3 params each

EPSILON = 1e-6


def main():
    lr = 0.001
    # AND
    w_and = initialize_weights(0, 10, 3)  # + bias at the end
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
    w_or = initialize_weights(0, 10, 3)  # and bias
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
    print("AND:", dataset_and[:, :-1] @ w_and[:-1] + w_and[-1])
    print("OR:", dataset_or[:, :-1] @ w_or[:-1] + w_or[-1])
    # AND: [-0.2500005  0.2499995  0.2499995  0.7499995]
    # OR: [0.2499995 0.7499995 0.7499995 1.2499995]
    # still not correct but closer, there is more freedom for the weights


if __name__ == "__main__":
    main()
