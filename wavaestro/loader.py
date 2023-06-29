from pandas import DataFrame
from glob import glob
from re import match
from mido import MidiFile
import pywt
import scipy.io.wavfile


def load_initial_dataset(path: str) -> DataFrame:
    lines = []
    for wave_file in glob(f"{path}/**/*.wav", recursive=True):
        base_file = match(r"(.*)\.wav", wave_file).group(1)
        mid = MidiFile(
            f"{base_file}.midi",
            clip=True,
        )

        sampling_frequency, signal = scipy.io.wavfile.read(wave_file)
        scales = (1, len(signal))

        # HERE
        coefficient, frequency = pywt.cwt(
            data=signal,
            scales=scales,
            wavelet="cgau2",
        )

        breakpoint()

        line = {
            "year": match(r".*\/(\d{4})\/.*", wave_file).group(1),
            "ticks_per_beat": mid.ticks_per_beat,
            "mid": mid,
            "sampling_frequency": sampling_frequency,
        }
        lines.append(line)

    df = DataFrame(lines)
    return df
