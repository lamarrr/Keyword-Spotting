from typing import Tuple

SAMPLE_RATE: int = 16000
BUFFER_INTERVAL: int = 250
INPUT_DURATION: int = 1000
LMFB_SHAPE: Tuple[int] = (36, 40, 1)
FREQ_MIN: float = 20
FREQ_MAX: float = 8000

X_MIN: float = -80.0
X_MAX: float = 0.0
SNR_RANGE = (-5, 10)
DEBUG_AUDIO_PER_CLASS = 200
BASE_DIR = "speech_commands"
MIN_FREQ = 20
MAX_FREQ = 8000

MFCC_SHAPE = (76, 40, 1)  # no longer
LMFB_SHAPE = (36, 40, 1)
# check for unknown
# number of augmentations for each

LEN_KNOWN = 2000
LEN_UNKNOWN = 12000

# TODO: Noise Folder must contain 16kHz, {DS_DUR} second minimum of wav files
SAMPLE_RATE = 16000
DS_DUR = 1000
# Background noise adapts to SNR ratio to Sound
# dB

NSAMPLES = int(SAMPLE_RATE * DS_DUR / 1000)
TARGET_KEYWORDS = ("marvin", "nora", "on")

categories = ("nora", "marvin", "on", "off", "down", "left", "right", "up",
              "stop", "unknown")

MODEL_WEIGHT_PATH = "nora/kws/assets/weights.epoch-143.loss-0.279.vloss-0.284.acc-0.907.vacc-0.909.hdf5"
