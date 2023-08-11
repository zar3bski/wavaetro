from wavaestro.loader import load_initial_dataset
import logging


logging.basicConfig(level=logging.INFO)


def main():
    df = load_initial_dataset(path="data/maestro-v3.0.0/2004", wavelet_name="sym13")


if __name__ == "__main__":
    main()
