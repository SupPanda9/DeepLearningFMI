from neuron import MLP
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engineering.value import draw_dot


def main() -> None:
    rng = np.random.default_rng(42)
    x = [2.0, 3.0, -1.0]

    n1 = MLP(in_channels=3, hidden_channels=[4, 4, 1], rng=rng)
    out = n1(x)
    print(out)

    n2 = MLP(in_channels=3, hidden_channels=[4, 4, 2], rng=rng)
    print(n2(x))

    draw_dot(out, fn="w02_p2_03_res").render(
        directory="./week02/modeling_neurons/graphviz_output", view=True
    )


if __name__ == "__main__":
    main()


# i cannot replicate the results from the test case
# mine Value(data=-0.6490834594731474)
# [Value(data=0.715728690853701), Value(data=0.6594519942431283)]
