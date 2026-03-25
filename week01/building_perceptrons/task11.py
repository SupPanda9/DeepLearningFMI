import numpy as np

EPSILON = 1e-6


class SquareModel:
    def __init__(self, lr=0.01):
        self.lr = lr
        n = 45

        self.w_hidden = np.zeros((n, 2))
        self.w_hidden[:, 0] = 1.0
        self.w_hidden[:, 1] = np.linspace(-1e-3, -1.73, n)

        self.w_hidden[0, 1] = 0.0
        self.w_output = np.random.randn(n + 1) * 1e-3
        self.w_output[0] = 0.0
        self.w_output[-1] = 0.0

        self.x_mean = 0.0
        self.x_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0

    def __call__(self, x):
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = self.forward(x_norm)
        return y_norm * self.y_std + self.y_mean

    def forward(self, x):
        x_val = abs(float(x))

        z = x_val * self.w_hidden[:, 0] + self.w_hidden[:, 1]
        h = self.relu(z)

        y = h @ self.w_output[:-1] + self.w_output[-1]
        return y

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def create_dataset(n):
        return np.array([(i, i**2) for i in range(-n, n + 1)])

    def forward_batch(self, X):
        X_abs = np.abs(X)  # |x|
        z = X_abs[:, None] * self.w_hidden[:, 0].T + self.w_hidden[:, 1]
        h = self.relu(z)
        y_norm = h @ self.w_output[:-1].T + self.w_output[-1]
        return y_norm

    def calculate_loss(self, dataset):
        x = dataset[:, 0]
        y = dataset[:, 1]
        pred = self.forward_batch(x)
        return np.mean((y - pred) ** 2)

    def train(self, dataset, epochs=50000):
        x_raw, y_raw = dataset[:, 0], dataset[:, 1]

        self.x_mean, self.x_std = x_raw.mean(), x_raw.std()
        self.y_mean, self.y_std = y_raw.mean(), y_raw.std()

        x_norm = (x_raw - self.x_mean) / self.x_std
        max_abs = np.max(np.abs(x_norm)) * 1.05
        self.w_hidden[:, 0] = 1.0
        thresholds = np.linspace(0, 1, self.w_hidden.shape[0]) ** 0.7 * max_abs
        self.w_hidden[:, 1] = -thresholds
        y_norm = (y_raw - self.y_mean) / self.y_std
        dataset = np.column_stack((x_norm, y_norm))

        for epoch in range(epochs):
            grad_hidden = np.zeros_like(self.w_hidden)
            grad_output = np.zeros_like(self.w_output)

            for i in range(self.w_hidden.shape[0]):
                for j in range(self.w_hidden.shape[1]):
                    self.w_hidden[i, j] += EPSILON
                    loss_plus = self.calculate_loss(dataset)

                    self.w_hidden[i, j] -= 2 * EPSILON
                    loss_minus = self.calculate_loss(dataset)

                    self.w_hidden[i, j] += EPSILON
                    grad_hidden[i, j] = (loss_plus - loss_minus) / (2 * EPSILON)

            for i in range(self.w_output.shape[0]):
                self.w_output[i] += EPSILON
                loss_plus = self.calculate_loss(dataset)

                self.w_output[i] -= 2 * EPSILON
                loss_minus = self.calculate_loss(dataset)

                self.w_output[i] += EPSILON
                grad_output[i] = (loss_plus - loss_minus) / (2 * EPSILON)

            self.w_hidden -= self.lr * grad_hidden
            self.w_output -= self.lr * grad_output

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {self.calculate_loss(dataset):.7f}")
