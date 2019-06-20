import wave

import librosa
import numpy as np
import pyaudio
import pydub
from pydub import AudioSegment
from scipy.io import wavfile
from tensorflow import keras
import autk
from . import config
from typing import Type

"""
def log_mel_energy(inputs: np.ndarray,
                   sr,
                   freq_min=20,
                   freq_max=8000,
                   n_fft=400,
                   stride=160,
                   n_mels=40) -> np.ndarray:
    specto = librosa.feature.melspectrogram(inputs,
                                            sr=sr,
                                            n_fft=n_fft,
                                            hop_length=stride,
                                            n_mels=n_mels,
                                            power=1,
                                            fmin=freq_min,
                                            fmax=freq_max)
    log_specto = librosa.core.amplitude_to_db(specto, ref=np.max)
    # Freq x Time -> Time x Freq
    log_specto = log_specto.T
    # TODO: change the line below in AUTK
    # log_specto = log_specto.reshape((1, *log_specto.shape))

    return log_specto


def int_samples_to_float(y: np.ndarray, dtype: Type):

    #assert isinstance(y, np.ndarray)
    #assert issubclass(y.dtype.type, np.integer)
    #assert issubclass(dtype, np.floating)

    y = y.astype(dtype) / np.iinfo(y.dtype).max

    return y
"""

def predict(model, in_data, frame_count):
    datanp = np.frombuffer(in_data, count=frame_count, dtype=np.int16)
    datanp = autk.int_samples_to_float(datanp, np.float32)
    lmfbe_input = autk.log_mel_energy(datanp,
                                 config.SAMPLE_RATE,
                                 n_mels=config.LMFB_SHAPE[1],
                                 n_fft=1024,
                                 stride=256 + 128 + 64)
    lmfbe_input = (lmfbe_input - config.X_MIN) / (config.X_MAX - config.X_MIN)
    lmfbe_input = lmfbe_input.reshape((1, *lmfbe_input.shape, 1))
    prediction = model.predict(lmfbe_input)
    catg = np.argmax(prediction.flatten())
    print("[KWS:Prediction] f(x|W...): %s" % config.categories[catg])
    return catg