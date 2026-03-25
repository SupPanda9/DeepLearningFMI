from utils import create_dataset, initialize_weights, calculate_loss, calculate_appr_derivative
import numpy as np

EPSILON = 1e-6

def main():
    np.random.seed(42)
    w = initialize_weights(0, 10)
    dataset = create_dataset(6)

    for _ in range(500):
        grad = calculate_appr_derivative(w, dataset, EPSILON)
        w -= 0.001 * grad

    print("\nAfter 500 updates with learning rate = 0.001, seed=42:", calculate_loss(w, dataset))
    print(f"And the weight is {w}")

    for i in range(3):
        w = initialize_weights(0, 10)
        dataset = create_dataset(6)
        for _ in range(500):
            grad = calculate_appr_derivative(w, dataset, EPSILON)
            w -= 0.001 * grad

        print(f"\n{[i]} After 500 updates with learning rate = 0.001:", calculate_loss(w, dataset))
        print(f"And the weight is {w}")
        
        # yes, it still converges


if __name__ == "__main__":
    main()
