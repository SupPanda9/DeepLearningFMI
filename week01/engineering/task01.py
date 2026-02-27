import numpy as np


def simulate_random_walk():
    rng = np.random.default_rng(123)

    random_float = rng.random()
    random_integer1 = rng.integers(1, 7)
    random_integer2 = rng.integers(1, 7)

    step = 50
    dice = rng.integers(1, 7)

    if dice <= 2:
        step -= 1
    elif dice <= 5:
        step += 1
    else:
        step += rng.integers(1, 7)

    return random_float, random_integer1, random_integer2, dice, step


def main():
    rf, r1, r2, dice, step = simulate_random_walk()

    print("Random float:", rf)
    print("Random integer 1:", r1)
    print("Random integer 2:", r2)
    print("Before throw step =", 50)
    print("After throw dice =", dice)
    print("After throw step =", step)


if __name__ == "__main__":
    main()