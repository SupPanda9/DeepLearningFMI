from utils import (
    simulate_clumsy_random_walk,
    simulate_multiple_random_walks,
    draw_endpoints,
)
import numpy as np


def main():
    all_walks = simulate_multiple_random_walks(500, 100, simulate_clumsy_random_walk)
    all_walks_endpoints = [walk[-1] for walk in all_walks]
    np_all_walks_endpoints = np.array(all_walks_endpoints)
    
    draw_endpoints(np_all_walks_endpoints)
    odds_over_60 = np.mean(np_all_walks_endpoints >= 60)
    print(odds_over_60)
    # odds of getting over 60 steps is 0.552

if __name__ == "__main__":
    main()
