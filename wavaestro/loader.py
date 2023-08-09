import logging
from pandas import DataFrame
from glob import glob
from re import match
from mido import MidiFile
import pywt
import scipy.io.wavfile
import matplotlib.pyplot as plt

from wavaestro.decorators import cached

logger = logging.getLogger(__name__)


# @cached("data/df")
def load_initial_dataset(path: str) -> DataFrame:
    lines = []
    for wave_file in glob(f"{path}/**/*.wav", recursive=True):
        logger.info("processing %s", wave_file)
        base_file = match(r"(.*)\.wav", wave_file).group(1)
        mid = MidiFile(
            f"{base_file}.midi",
            clip=True,
        )

        sampling_frequency, signal = scipy.io.wavfile.read(wave_file)

        _, db1 = pywt.dwt(signal, "db1", axis=0)
        _, bior13 = pywt.dwt(signal, "bior1.3", axis=0)
        _, bior37 = pywt.dwt(signal, "bior3.7", axis=0)
        _, sym13 = pywt.dwt(signal, "sym13", axis=0)

        line = {
            "year": match(r".*\/(\d{4})\/.*", wave_file).group(1),
            "ticks_per_beat": mid.ticks_per_beat,
            "mid": mid,
            "sampling_frequency": sampling_frequency,
            "signal": signal,
            "db1": db1,
            "bior13": bior13,
            "bior37": bior37,
            "sym13": sym13,
        }
        lines.append(line)

    df = DataFrame(lines)
    return df
