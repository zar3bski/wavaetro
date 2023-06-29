from pandas import DataFrame
from glob import glob
from re import match
from mido import MidiFile


def load_initial_dataset(path: str) -> DataFrame:
    lines = []
    for wave_file in glob(f"{path}/**/*.wav", recursive=True):
        base_file = match(r"(.*)\.wav", wave_file).group(1)
        mid = MidiFile(
            f"{base_file}.midi",
            clip=True,
        )
        line = {
            "year": match(r".*\/(\d{4})\/.*", wave_file).group(1),
            "ticks_per_beat": mid.ticks_per_beat,
            "mid": mid,
        }
        lines.append(line)

    df = DataFrame(lines)
    return df
