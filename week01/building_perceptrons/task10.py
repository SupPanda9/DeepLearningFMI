import numpy as np
from utils import (
    create_dataset_xor,
    sigmoid,
)

EPSILON = 1e-6


class Xor:
    def __init__(self, lr=0.05):
        self.lr = lr
        # we know that xor = (or) and (nand) so we can reuse previous models
        # or we can use that structure to train the model
        # we need more hidden layers with 9 parameters total
        self.w_hidden = np.random.uniform(
            -1, 1, (2, 3)
        )  # weights for OR + bias and NAND + bias
        self.w_output = np.random.uniform(-1, 1, 3)  # weights for AND + bias

    def forward(self, x1, x2 = None):
        if x2 is not None:
            x = np.array([x1, x2])
        else:
            x = x1

        z_hidden = x @ self.w_hidden[:, :-1] + self.w_hidden[:, -1]
        a_hidden = sigmoid(z_hidden)

        z_output = a_hidden @ self.w_output[:-1] + self.w_output[-1]
        a_output = sigmoid(z_output)
        return a_output

    def calculate_loss(self, dataset):
        x = dataset[:, :-1]
        y = dataset[:, -1]
        prediction = self.forward(x)

        return np.mean((y - prediction) ** 2)

    def train(self, dataset, epochs=100000):
        for epoch in range(epochs + 1):
            grad_hidden = np.zeros_like(self.w_hidden)
            grad_output = np.zeros_like(self.w_output)

            for i in range(self.w_hidden.shape[0]):
                for j in range(self.w_hidden.shape[1]):
                    original = self.w_hidden[i, j]
                    self.w_hidden[i, j] += EPSILON
                    loss_eps = self.calculate_loss(dataset)

                    self.w_hidden[i, j] = original
                    loss = self.calculate_loss(dataset)
                    grad_hidden[i, j] = (loss_eps - loss) / EPSILON

            for i in range(self.w_output.shape[0]):
                original = self.w_output[i]
                self.w_output[i] += EPSILON
                loss_eps = self.calculate_loss(dataset)

                self.w_output[i] = original
                loss = self.calculate_loss(dataset)
                grad_output[i] = (loss_eps - loss) / EPSILON

            self.w_hidden -= self.lr * grad_hidden
            self.w_output -= self.lr * grad_output

            if epoch % 10000 == 0:
                current_loss = self.calculate_loss(dataset)
                print(f"Epoch {epoch}, Loss: {current_loss:.6f}")


if __name__ == "__main__":
    dataset_xor = create_dataset_xor()

    xor_model = Xor(lr=0.05)
    xor_model.train(dataset_xor, epochs=100000)

    for x1, x2, y in dataset_xor:
        y_pred = xor_model.forward(x1, x2)
        print(f"XOR({x1}, {x2}) = {y_pred:.4f}")