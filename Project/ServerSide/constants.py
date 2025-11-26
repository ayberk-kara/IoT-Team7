from pathlib import Path

FRAME_SIZE = 7
SAMPLING_RATE = 100
FLOOR_RANGES = [
    (-10.0, 2.5, "Ground"),
    (2.5, 7.5, "First"),
    (7.5, 15.0, "Second"),
]
FLATLINE_STD_DEV_THRESHOLD = 0.05
FALL_HEIGHT_THRESHOLD = 0.4
PEOPLE_DIR = Path(__file__).parent / 'people'
CACHE_DIR = "cache"