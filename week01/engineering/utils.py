import numpy as np
import matplotlib.pyplot as plt

CLUMSINESS_FACTOR = 0.005
DICE_SIDES = 6
RETURN_DICE_SCORE = 2
CLIMB_DICE_SCORE = 5


def simulate_random_walk(n: int, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)

    step = 0
    steps = [step]

    for _ in range(0, n):
        dice = rng.integers(1, DICE_SIDES + 1)
        if dice <= RETURN_DICE_SCORE:
            step = max(0, step - 1)
        elif dice <= CLIMB_DICE_SCORE:
            step += 1
        else:
            step += rng.integers(1, DICE_SIDES + 1)
        steps.append(int(step))

    return steps


def simulate_clumsy_random_walk(n: int, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)

    step = 0
    steps = [step]

    for _ in range(0, n):
        dice = rng.integers(1, DICE_SIDES + 1)
        if dice <= RETURN_DICE_SCORE:
            step = max(0, step - 1)
        elif dice <= CLIMB_DICE_SCORE:
            step += 1
        else:
            step += rng.integers(1, DICE_SIDES + 1)

        clumsiness = rng.random()
        if clumsiness <= CLUMSINESS_FACTOR:
            step = 0

        steps.append(int(step))

    return steps


def simulate_multiple_random_walks(
    num_walks, num_throws, random_walk=simulate_random_walk
):
    rng = np.random.default_rng(123)
    all_walks = []
    for _ in range(0, num_walks):
        steps = random_walk(num_throws, rng)
        all_walks.append(steps)
    return all_walks


def draw_walks(all_walks: np.array):
    plt.plot(all_walks.T)
    plt.title("Random_walks")
    plt.xlabel("Throw")
    plt.show()


def draw_endpoints(all_walks_endpoints: np.array): 
    plt.hist(all_walks_endpoints)
    ticks = list(range(0, 121, 20))
    plt.xticks(ticks)
    plt.title("Random_walks")
    plt.xlabel("End step")
    plt.show()
