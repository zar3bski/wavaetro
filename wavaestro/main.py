from wavaestro.loader import load_initial_dataset
import logging


logging.basicConfig(level=logging.INFO)


def main():
    df = load_initial_dataset("data/maestro-v3.0.0/2004")


if __name__ == "__main__":
    main()
