import pickle


RESULTS_PATH = "/project/def-lulam50//results.pck"


def read_results():
    with open(RESULTS_PATH, "rb") as f:
        return pickle.load(f)


def main():
    results = read_results()

    print("hih")


if __name__ == "__main__":
    main()
