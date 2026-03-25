from utils import simulate_multiple_random_walks


def main():
    all_walks = simulate_multiple_random_walks(5, 100)
    print(all_walks)


if __name__ == "__main__":
    main()