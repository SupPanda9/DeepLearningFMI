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
        for p in n.parameters():
             p.grad = 0.0

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