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
    
    epochs = 10
    lr = 0.1

    for _ in range(epochs):
        # for p in n.parameters():
        #      p.grad = 0.0

        first_w = n.layers[0].neurons[0].w[0]
        print(f"Gradient of the first weight in the first neuron in the first layer: {first_w.grad}")

        y_preds = [n(x) for x in xs]
        loss = sum([(ypr - yt)**2 for yt, ypr in zip(ys, y_preds)])
        loss.backward()

        for p in n.parameters():
            p.data -= lr * p.grad
            
        print(f"Loss={loss.data}")

    final_preds = [n(x).data for x in xs]
    print("Predictions:")
    print(final_preds)

if __name__ == "__main__":
    main()

# 1. What do you notice about the gradient of the first weight
# in the first neuron in the first layer?
# Not zeroing the gradients.

# 2. Why is this a problem?
# If the bug was forgetting to zero gradients, it would be a problem
# because gradients would accumulate across epochs — each backward() call
# would add on top of the previous one instead of replacing it, causing the
# gradient to grow without bounds and the updates to blow up,
# making the model diverge rather than converge.