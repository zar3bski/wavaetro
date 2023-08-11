from mido import MidiFile

from wavaestro.loader import load_initial_dataset


def test_load():
    df = load_initial_dataset("test/data/maestro-v3.0.0", "db1")
    assert len(df) == 2
