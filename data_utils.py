import torch
import torchaudio
import tensorflow as tf

from utils import EPSILON, safe_div
from transforms import mask


def load_wav(wav_fname: str):
    '''
    OUTPUT
    complex_specs: list of complex spectrograms
                   each complex spectrogram has shape of
                   [freq, time, chan*2]
    '''

    stft = torchaudio.transforms.Spectrogram(512, power=None)

    wav, r = torchaudio.load(wav_fname)
    wav = torchaudio.compliance.kaldi.resample_waveform(
        wav, r, 16000)
    wav = normalize(wav)
    wav = stft(wav)

    # [chan, freq, time, 2] -> [freq, time, chan, 2]
    wav = wav.numpy().transpose(1, 2, 3, 0)
    wav = wav.reshape((*wav.shape[:2], -1))

    return wav


def normalize(wav):
    rms = torch.sqrt(torch.mean(torch.pow(wav, 2))) * 10
    return wav / rms


def minmax(x, y=None):
    # batch-wise pre-processing
    axis = tuple(range(1, len(x.shape)))

    # MIN-MAX
    x_max = tf.math.reduce_max(x, axis=axis, keepdims=True)
    x_min = tf.math.reduce_min(x, axis=axis, keepdims=True)
    x = safe_div(x-x_min, x_max-x_min)
    if y is not None:
        return x, y
    return x


def log_on_mel(mel, labels=None):
    mel = tf.math.log(mel + EPSILON)

    if labels is not None:
        return mel, labels
    return mel


def augment(specs, labels, time_axis=-2, freq_axis=-3):
    specs = mask(specs, axis=time_axis, max_mask_size=24, n_mask=6)
    specs = mask(specs, axis=freq_axis, max_mask_size=16)
    return specs, labels


def to_frame_labels(x, y):
    """
    :param y: [..., n_voices, n_frames, n_classes]
    :return: [..., n_frames, n_classes]
    """
    y = tf.reduce_sum(y, axis=-3)
    return x, y


def mono_chan(x, y):
    return x[..., :1], y


def stereo_mono(x, y=None):
    if y is None:
        return tf.concat([x[..., :2], x[..., :1] + x[..., 1:2], x[..., 2:4], x[..., 2:3] + x[..., 3:4]], -1)
    return tf.concat([x[..., :2], x[..., :1] + x[..., 1:2], x[..., 2:4], x[..., 2:3] + x[..., 3:4]], -1), y


def label_downsample(resolution=32):
    def _label_downsample(x, y):
        if isinstance(y, (list, tuple)):
            y_ = y[0]
            y_ = tf.keras.layers.AveragePooling1D(resolution, resolution, padding='same')(y_)
            y_ = tf.cast(y_ >= 0.5, y_.dtype)[:resolution]
            y = (y_,) + tuple([*y[1:]])
        else:
            y = tf.keras.layers.AveragePooling1D(resolution, resolution, padding='same')(y)
            y = tf.cast(y >= 0.5, y.dtype)[:resolution]

        return x, y
    return _label_downsample


def random_merge_aug(number):
    def _random_merge_aug(x, y=None):
        chan = x.shape[-1] // 2
        if chan != 2:
            raise ValueError('This augment can be used in 2 channel audio')

        real = x[...,:chan]
        imag = x[...,chan:]
        
        factor = tf.random.uniform((1, 1, number - chan), 0.1, 0.9)
        aug_real = factor * tf.repeat(real[..., :1], number - chan, -1) + tf.sqrt(1 - factor) * tf.repeat(real[..., 1:], number - chan, -1)
        
        real = tf.concat([real, aug_real], -1)
        imag = tf.concat([imag, tf.repeat(imag[...,:1] + imag[...,1:], number - chan, -1)], -1)
        if y is not None:
            return tf.concat([real, imag], -1), y
        return tf.concat([real, imag], -1)
    return _random_merge_aug


def multiply_label(multiply_factor):
    def _multiply_label(x, y):
        return x, y * multiply_factor
    return _multiply_label


def stft_filter(filter_num):
    def _stft_filter(x, y=None):
        mask = tf.concat([tf.ones([1] + [*x.shape[1:]], x.dtype),
                          tf.zeros([filter_num] + [*x.shape[1:]], x.dtype),
                          tf.ones([x.shape[0] - filter_num - 1] + [*x.shape[1:]], x.dtype),
                          ], 0)
        x *= mask
        if y is None:
            return x
        return x, y
    return _stft_filter


def speech_enhancement_preprocess(x, y=None):
    """
    :param y: ([..., n_voices, n_frames, n_classes], ..., ...)
    :return: [..., n_frames, n_classes]
    """
    x = x[1:,...,:x.shape[-1] // 2]
    if y is None:
        return x
    y = (tf.reduce_sum(y[0], axis=-3), y[1][1:, ...,:x.shape[-1] // 2], y[2][1:, ...,:x.shape[-1] // 2])
    return x, y


if __name__ == '__main__':
    import glob
    wavs = glob.glob('/codes/2020_track3/t3_audio/*.wav')
    print(wavs)
    stfts = [load_wav(wav) for wav in wavs]

    for stft in stfts:
        print(stft.shape)

