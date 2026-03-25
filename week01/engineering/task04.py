from task03 import simulate_random_walk
import matplotlib.pyplot as plt


def draw_walk():
    steps = simulate_random_walk(100)
    steps.pop(0)

    plt.plot(steps)
    plt.title("Random Walk")
    plt.xlabel("Throw")
    plt.tight_layout()
    plt.show()


def main():
    draw_walk()


if __name__ == "__main__":
    main()


# the plots are different but values are the same?!