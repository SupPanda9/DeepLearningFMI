import numpy as np


def simulate_random_walk(n: int):
    rng = np.random.default_rng(123)

    step = 0
    steps = [step]

    for _ in range(0, n):
        dice = rng.integers(1, 7)
        if dice <= 2:
            step -= 1
        elif dice <= 5:
            step += 1
        else:
            step += rng.integers(1, 7)
        steps.append(int(step))

    return steps


def main():
    steps = simulate_random_walk(100)
    print(steps)


if __name__ == "__main__":
    main()

# In the beginning there is a negative value -
# impossible when counting steps
