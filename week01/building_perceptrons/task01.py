import numpy as np

# y = w * x it should have one parameter w (which should become something close to 2)


def create_dataset(n: int):
    return [(i, i * 2) for i in range(0, n)]


def initialize_weights(x: int, y: int):
    return np.random.uniform(x, y)


def main():
    print(create_dataset(4))
    print(initialize_weights(0, 100))
    print(initialize_weights(0, 10))


if __name__ == "__main__":
    main()
