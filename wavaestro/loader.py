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

        _, haar = pywt.dwt(data=signal, wavelet="haar")
        _, bior39 = pywt.dwt(data=signal, wavelet="bior3.9")

        line = {
            "year": match(r".*\/(\d{4})\/.*", wave_file).group(1),
            "ticks_per_beat": mid.ticks_per_beat,
            "mid": mid,
            "sampling_frequency": sampling_frequency,
            "signal": signal,
            "haar": haar,
            "bior39": bior39,
        }
        lines.append(line)

    df = DataFrame(lines)
    return df
