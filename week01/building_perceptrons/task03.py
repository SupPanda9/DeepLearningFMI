from utils import create_dataset, initialize_weights, calculate_loss
import numpy as np

EPSILON = 1e-6


def calculate_appr_derivative(w: float, dataset: np.ndarray, eps: float):
    return (calculate_loss(w + eps, dataset) - calculate_loss(w, dataset)) / eps


def main():
    np.random.seed(42)
    w = initialize_weights(0, 10)
    dataset = create_dataset(6)

    print("Before updating:", calculate_loss(w, dataset))

    grad = calculate_appr_derivative(w, dataset, EPSILON)
    w -= grad

    print("After updating:", calculate_loss(w, dataset))

    # with these values we can see oscilation ()
    for learning_rate in [0.1, 0.01, 0.001]:
        np.random.seed(42)
        w = initialize_weights(0, 10)
        print(f"\nLearning rate: {learning_rate}")
        print("Before updating:", calculate_loss(w, dataset))

        grad = calculate_appr_derivative(w, dataset, EPSILON)
        w -= learning_rate * grad

        print("After updating:", calculate_loss(w, dataset))

    np.random.seed(42)
    w = initialize_weights(0, 10)
    dataset = create_dataset(6)
    for _ in range(10):
        grad = calculate_appr_derivative(w, dataset, EPSILON)
        w -= 0.001 * grad

    print("\nAfter 10 updates with learning rate = 0.001:", calculate_loss(w, dataset))


if __name__ == "__main__":
    main()
