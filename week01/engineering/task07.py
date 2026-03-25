from utils import (
    simulate_clumsy_random_walk,
    simulate_multiple_random_walks,
    draw_walks,
)
import numpy as np


def main():
    all_walks = simulate_multiple_random_walks(20, 100, simulate_clumsy_random_walk)
    np_all_walks = np.array(all_walks)
    draw_walks(np_all_walks)


if __name__ == "__main__":
    main()
