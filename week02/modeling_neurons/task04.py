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
    print(f"{y_preds=}")

    loss = sum([(ypr - yt) ** 2 for yt, ypr in zip(ys, y_preds)])
    print(f"Loss = {loss}")

    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    first_weight_grad = n.layers[0].neurons[0].w[0].grad

    print(
        f"Gradient of the first weight in the first neuron in the first layer: {first_weight_grad}."
    )

    print(f"Parameters: {n.parameters()}")


if __name__ == "__main__":
    main()

# only 36 params in the expected output?
# but the wanted architecture has 41 params - 3 inputs + bias for 4 neurons = 16 weights
# 4 neurons + bias for each to 4 neurons = 20 weights
# 4 neurons + bias for each to 1 neuron = 5 weights
# = 41

# the same problems with randomness as before
