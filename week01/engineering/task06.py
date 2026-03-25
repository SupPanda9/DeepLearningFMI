from utils import simulate_multiple_random_walks, draw_walks
import numpy as np


def main():
    all_walks = simulate_multiple_random_walks(5, 100)
    all_walks = np.array(all_walks)
    draw_walks(all_walks)


if __name__ == "__main__":
    main()
