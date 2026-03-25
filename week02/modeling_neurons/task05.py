from neuron import MLP
import numpy as np


def main() -> None:
    np.random.seed(42)
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    n = MLP(in_channels=3, hidden_channels=[4, 4, 1])
    y_preds = [n(x) for x in xs]
    loss = sum([(ypr - yt) ** 2 for yt, ypr in zip(ys, y_preds)])
    print(f"Before gradient step: {loss}")

    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.1
    for p in n.parameters():
        p.data -= lr * p.grad

    y_preds_after = [n(x) for x in xs]
    loss_after = sum([(ypr - yt) ** 2 for yt, ypr in zip(ys, y_preds_after)])
    print(f"After gradient step: {loss_after}.")


if __name__ == "__main__":
    main()
