import numpy as np 
from utils import sigmoid
import matplotlib.pyplot as plt

def main():
    x = np.arange(-10, 10.1, 0.5)
    y = sigmoid(x)

    plt.plot(x, y)
    plt.title("Sigmoid function")
    plt.show()


if __name__ == "__main__":
    main()