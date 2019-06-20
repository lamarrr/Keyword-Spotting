import enum
import gc
import glob
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, Type

import librosa
import numpy as np
import tqdm
from pydub import AudioSegment
from scipy.io import wavfile


from config import *

if __name__ == "__main__":
    assert len(sys.argv) == 2
    DEBUG_DIR = sys.argv[1]

SOURCE_DIR = "sorted"
catg_fs = [
    "marvin", "on", "off", "down", "left", "right", "up", "stop", "unknown",
    "nora"
]
catg_indexes: Dict[str, int] = {
    "nora": 0,
    "marvin": 1,
    "on": 2,
    "off": 3,
    "down": 4,
    "left": 5,
    "right": 6,
    "up": 7,
    "stop": 8,
    "unknown": 9
}


# Write your own boilerplate
def _Q(n: int) -> int:
    return int(np.ceil(LEN_KNOWN / n))


def _G(n: int) -> int:
    return int(np.ceil(LEN_UNKNOWN / n))


catg_multiples = {
    "marvin": _Q(1746),
    "on": _Q(2367),
    "off": _Q(2357),
    "down": _Q(2359),
    "left": _Q(2353),
    "right": _Q(2367),
    "up": _Q(2375),
    "stop": _Q(2380),
    "unknown": _G(12000),
    "nora": _Q(125)
}

background_noise_folder = "background_noise"

if __name__ == "__main__":

    background_noise_paths = glob.glob(
        os.path.join(BASE_DIR, SOURCE_DIR, background_noise_folder, "*.wav"))

else:
    background_noise_paths = glob.glob("test_suite/backgrounds/*.wav")

assert len(background_noise_paths) > 0


def int_samples_to_float(y: np.ndarray, dtype: Type):

    #assert isinstance(y, np.ndarray)
    #assert issubclass(y.dtype.type, np.integer)
    #assert issubclass(dtype, np.floating)

    y = y.astype(dtype) / np.iinfo(y.dtype).max

    return y


def float_samples_to_int(y: np.ndarray, dtype: Type):

    #assert isinstance(y, np.ndarray)
    #assert issubclass(y.dtype.type, np.floating)
    #assert issubclass(dtype, np.integer)

    return (y * np.iinfo(dtype).max).astype(dtype)


def log_mel_energy(inputs: np.ndarray, sr, n_fft=400, stride=160,
                   n_mels=40) -> np.ndarray:
    specto = librosa.feature.melspectrogram(inputs,
                                            sr=sr,
                                            n_fft=n_fft,
                                            hop_length=stride,
                                            n_mels=n_mels,
                                            power=1,
                                            fmin=MIN_FREQ,
                                            fmax=MAX_FREQ)
    log_specto = librosa.core.amplitude_to_db(specto, ref=np.max)

    # R -> Time x Freq
    log_specto = log_specto.T
    log_specto = log_specto.reshape((1, *LMFB_SHAPE))

    return log_specto


def fetch_audio(wav_path: str) -> np.ndarray:
    # Process then save to TARGET_BASE_DIR
    audio = AudioSegment.from_wav(wav_path)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_channels(1)
    audio = audio[:DS_DUR]
    padding = AudioSegment.silent(duration=DS_DUR, frame_rate=SAMPLE_RATE)
    audio = padding.overlay(audio)
    audio.export(wav_path, format="wav")

    dBFS = audio.dBFS
    audio, sample_rate = librosa.core.load(wav_path, sr=SAMPLE_RATE, mono=True)

    return audio, dBFS


def tx_mfcc(data: np.ndarray,
            sample_rate: int = SAMPLE_RATE,
            n_mfcc: int = 40,
            stride: int = 20,
            window_size: int = 40):
    """
    data - np.float32 ndarray (n,)
    stride - ms
    window_size - ms
    """

    #assert isinstance(data, np.ndarray)
    #assert isinstance(n_mfcc, int)
    #assert isinstance(sample_rate, int)
    #assert isinstance(stride, int)
    #assert isinstance(window_size, int)
    #assert data.dtype.type == np.float32

    stride = int(sample_rate * stride / 1000)
    window_size = int(sample_rate * window_size / 1000)

    result: np.ndarray = librosa.feature.mfcc(y=data,
                                              sr=sample_rate,
                                              n_mfcc=n_mfcc,
                                              hop_length=stride,
                                              n_fft=window_size).astype(
                                                  np.float32)

    # Features x Time > Time x Features

    result = result.T

    return result.reshape((1, *MFCC_SHAPE))


def preload_background_noise(path: str) -> AudioSegment:
    audio = AudioSegment.from_wav(path)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    assert int(
        audio.duration_seconds * 1000
    ) >= DS_DUR, "Background Noise shorter than specified Dataset Duration"
    return audio


# Crop, Set frame rate and Duration
background_noises = tuple(
    preload_background_noise(path) for path in background_noise_paths)


def shuffle_unison(arg: Iterable[Any], *other_args: Iterable[Iterable[Any]]):

    args = (arg, *other_args)
    state = np.random.get_state()
    for argument in args:
        np.random.set_state(state)
        np.random.shuffle(argument)

    return None


def random_clip(audio: AudioSegment, target_duration: int) -> AudioSegment:
    aud_dur = int(audio.duration_seconds * 1000)
    #assert aud_dur >= target_duration
    begin = random.randint(0, aud_dur - target_duration)
    return audio[begin:begin + target_duration]


def background_noise_augment(y_overlay: np.ndarray, dBFS: float) -> np.ndarray:

    #assert isinstance(y_overlay, np.ndarray)
    #assert issubclass(y_overlay.dtype.type, np.floating)

    # Select within range
    snr = random.random()
    snr *= (SNR_RANGE[1] - SNR_RANGE[0])
    target_noise_dBFS = dBFS - snr

    bg_noise = random.choice(background_noises)

    gain = target_noise_dBFS - bg_noise.dBFS
    bg_noise = bg_noise.apply_gain(gain)
    bg_noise = random_clip(bg_noise, DS_DUR)
    bg_noise = np.array(bg_noise.get_array_of_samples())

    #assert bg_noise.dtype == np.int16

    bg_noise = int_samples_to_float(bg_noise, np.float32)

    return y_overlay + bg_noise


def speed_augment(y_speed: np.ndarray, dBFS: float) -> np.ndarray:

    #assert isinstance(y_speed, np.ndarray)
    #assert issubclass(y_speed.dtype.type, np.floating)
    #assert isinstance(dBFS, float)

    speed_change = np.random.uniform(low=0.9, high=1.1)
    tmp = librosa.effects.time_stretch(y_speed, speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed = np.zeros_like(y_speed)
    y_speed[0:minlen] = tmp[0:minlen]

    return y_speed


# Model is trained under noisy conditions
def white_noise_augment(y_noise: np.ndarray, dBFS: float) -> np.ndarray:

    # dBFS
    #assert isinstance(y_noise, np.ndarray)
    #assert issubclass(y_noise.dtype.type, np.floating)
    #assert isinstance(dBFS, float)

    noise_amp = 0.005 * np.random.uniform() * np.amax(y_noise)
    y_noise = y_noise + (noise_amp * np.random.normal(size=y_noise.shape[0]))

    return y_noise.astype(np.float32)


def pitch_augment(y_pitch: np.ndarray, dBFS: float) -> np.ndarray:

    #assert isinstance(y_pitch, np.ndarray)
    #assert issubclass(y_pitch.dtype.type, np.floating)
    #assert isinstance(dBFS, float)

    bins_per_octave = 24
    pitch_pm = 4
    pitch_change = pitch_pm * 2 * (np.random.uniform() - 0.5)
    y_pitch = librosa.effects.pitch_shift(y_pitch,
                                          SAMPLE_RATE,
                                          n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)

    return y_pitch


def value_augment(y_aug: np.ndarray, dBFS: float) -> np.ndarray:

    #assert isinstance(y_aug, np.ndarray)
    #assert issubclass(y_aug.dtype.type, np.floating)
    #assert isinstance(dBFS, float)

    dyn_change = np.random.uniform(low=0.5, high=1.1)
    y_aug = y_aug * dyn_change

    return y_aug


def random_shift_augment(y_shift: AudioSegment, dBFS: float) -> np.ndarray:

    #assert isinstance(y_shift, np.ndarray)
    #assert issubclass(y_shift.dtype.type, np.floating)
    #assert isinstance(dBFS, float)

    timeshift_fac = 0.2 * 2 * (np.random.uniform() - 0.5
                               )  # up to 20% of length
    start = int(y_shift.shape[0] * timeshift_fac)
    if (start > 0):
        y_shift = np.pad(y_shift, (start, 0),
                         mode="constant")[0:y_shift.shape[0]]
    else:
        y_shift = np.pad(y_shift, (0, -start),
                         mode="constant")[0:y_shift.shape[0]]

    return y_shift


def hpss_augment(y_hpss: np.ndarray, dBFS: float) -> np.ndarray:

    #assert isinstance(y_hpss, np.ndarray)
    #assert issubclass(y_hpss.dtype.type, np.floating)
    #assert isinstance(dBFS, float)

    y_hpss = librosa.effects.hpss(y_hpss)

    return y_hpss[1]


def pitch_speed_augment(y_pitch_speed: np.ndarray, dBFS: float) -> np.ndarray:

    #assert isinstance(y_pitch_speed, np.ndarray)
    #assert issubclass(y_pitch_speed.dtype.type, np.floating)
    #assert isinstance(dBFS, float)

    length_change = np.random.uniform(low=0.5, high=1.5)
    speed_fac = 1.0 / length_change
    tmp = np.interp(np.arange(0, len(y_pitch_speed), speed_fac),
                    np.arange(0, len(y_pitch_speed)), y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed = np.zeros_like(y_pitch_speed)
    y_pitch_speed[0:minlen] = tmp[0:minlen]

    return y_pitch_speed


def none_augment(y_none: np.ndarray, dBFS: float) -> np.ndarray:

    #assert isinstance(y_none, np.ndarray)
    #assert issubclass(y_none.dtype.type, np.floating)
    #assert isinstance(dBFS, float)

    return np.copy(y_none)


__augmentations__ = (white_noise_augment, random_shift_augment, pitch_augment)
__aug_suffixes__ = ("wna", "rsa", "pa")
__aug_priorites__ = (16, 16, 2)
__const_eff_aug_indexes__ = tuple()


# contains logic for choice of augmentation
class AugChoice:
    def __init__(self):
        self.choices = []
        for choice, priority in zip(range(len(__augmentations__)),
                                    __aug_priorites__):
            self.choices.extend((choice, ) * priority)

    def get_aug(self):
        choice_count = len(self.choices)
        choice = random.choice(self.choices)

        if choice in __const_eff_aug_indexes__:
            while True:
                try:
                    self.choices.remove(choice)
                except:
                    break

        return __augmentations__[choice], __aug_suffixes__[choice]


def generate_augmentations(audio: np.ndarray, dBFS: float,
                           n_augmentations: int):

    # Check all assumptions
    #assert audio.size == NSAMPLES
    #assert type(dBFS) == float
    #assert isinstance(audio, np.ndarray)
    #assert isinstance(dBFS, float)
    #assert isinstance(n_augmentations, int)

    aug_choice = AugChoice()

    augmentations = []
    augmentation_names = []

    for _ in range(n_augmentations):
        augmentation, suffix = aug_choice.get_aug()
        aug = augmentation(audio, dBFS)
        aug = background_noise_augment(aug, dBFS)
        augmentations.append(aug)
        augmentation_names.append(suffix)

    return augmentations, augmentation_names


def main():
    # run relative
    TARGET_BASE_PATH = os.path.join(BASE_DIR, DEBUG_DIR)

    os.mkdir(TARGET_BASE_PATH)
    #X = np.ndarray((0, *LMFB_SHAPE), dtype=np.float32)
    #Y = np.ndarray((0, ), dtype=np.int32)

    for catg_f in catg_fs:

        C_DEBUG_DIR = os.path.join(TARGET_BASE_PATH, catg_f)
        os.mkdir(C_DEBUG_DIR)

        print("Generating for class:", catg_f)
        wav_paths = glob.glob(
            os.path.join(BASE_DIR, SOURCE_DIR, catg_f, "*.wav"))

        assert len(wav_paths) > 0, "Folder contains no wav files"

        # class specific
        X_c_spec = np.ndarray((
            0,
            *LMFB_SHAPE,
        ), dtype=np.float32)
        Y_c_spec = np.ndarray((0, ), dtype=np.int32)

        ndebugs = 0

        for wav_path in tqdm.tqdm(wav_paths):

            audio, dBFS = fetch_audio(wav_path)

            if catg_f == "unknown":
                if random.random() > 0.8:
                    # empty data points always with noise
                    audio, dBFS = np.zeros_like(audio, dtype=audio.dtype), dBFS

            datapoints, aug_names = generate_augmentations(
                audio, dBFS, catg_multiples[catg_f])

            for datapoint, aug_name in zip(datapoints, aug_names):

                lmfbe = log_mel_energy(datapoint,
                                       SAMPLE_RATE,
                                       n_mels=LMFB_SHAPE[1],
                                       n_fft=1024,
                                       stride=256 + 128 + 64)

                if catg_f != "unknown":
                    assert not np.all(lmfbe == 0)
                X_c_spec = np.append(X_c_spec, lmfbe, axis=0)
                Y_c_spec = np.append(Y_c_spec, catg_indexes[catg_f])

                gc.collect()

                if ndebugs < DEBUG_AUDIO_PER_CLASS:
                    filename = "%s_%d_%s.wav" % (catg_f, ndebugs + 1, aug_name)
                    path = os.path.join(C_DEBUG_DIR, filename)
                    wavfile.write(filename=path,
                                  rate=SAMPLE_RATE,
                                  data=datapoint)
                    ndebugs += 1

        shuffle_unison(X_c_spec, Y_c_spec)

        length = LEN_KNOWN if catg_f != "unknown" else LEN_UNKNOWN
        X_c_spec = X_c_spec[:length]
        Y_c_spec = Y_c_spec[:length]

        np.save(os.path.join(BASE_DIR, DEBUG_DIR, "X_%s.npy" % catg_f),
                X_c_spec)
        np.save(os.path.join(BASE_DIR, DEBUG_DIR, "Y_%s.npy" % catg_f),
                Y_c_spec)
        #X = np.append(X, X_c_spec, axis=0)
        #Y = np.append(Y, Y_c_spec)
        gc.collect()

    #shuffle_unison(X, Y)
    #np.save(os.path.join(BASE_DIR, DEBUG_DIR, "X.npy"), X)
    #np.save(os.path.join(BASE_DIR, DEBUG_DIR, "Y.npy"), Y)


if __name__ == "__main__":
    main()
