import numpy as np
import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE
EPSILON = 1e-8
LOG_EPSILON = tf.math.log(EPSILON)


""" FEATURE INDEPENDENT AUGMENTATIONS """
def mask(specs, axis, max_mask_size=None, n_mask=1):
    def make_shape(size):
        # returns (1, ..., size, ..., 1)
        shape = [1] * len(specs.shape)
        shape[axis] = size
        return tuple(shape)

    total = specs.shape[axis]
    mask = tf.ones(make_shape(total), dtype=specs.dtype)
    if max_mask_size is None:
        max_mask_size = total

    for i in range(n_mask):
        size = tf.random.uniform([], maxval=max_mask_size, dtype=tf.int32)
        offset = tf.random.uniform([], maxval=total-size, dtype=tf.int32)

        mask *= tf.concat(
            (tf.ones(shape=make_shape(offset), dtype=mask.dtype),
             tf.zeros(shape=make_shape(size), dtype=mask.dtype),
             tf.ones(shape=make_shape(total-size-offset), dtype=mask.dtype)),
            axis=axis)

    return specs * mask


def random_shift(specs, axis=0, width=16):
    new_specs = tf.pad(specs, [[0]*2 if i != axis else [width]*2
                               for i in range(len(specs.shape))])
    new_specs = tf.image.random_crop(new_specs, specs.shape)
    return new_specs


""" MAGNITUDE-PHASE SPECTROGRAM """
def magphase_mixup(alpha=2., feat='magphase'):
    '''
    returns magphase
    '''
    assert feat in ['magphase', 'complex']
    import tensorflow_probability as tfp
    beta = tfp.distributions.Beta(alpha, alpha)

    def _mixup(specs, labels):
        # preprocessing
        specs = tf.cast(specs, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.float32)

        indices = tf.reduce_mean(
            tf.ones_like(labels, dtype=tf.int32),
            axis=range(1, len(labels.shape)))
        indices = tf.cumsum(indices, exclusive=True)
        indices = tf.random.shuffle(indices)

        # assume mag, phase...
        if feat == 'magphase':
            specs = magphase_to_complex(specs)
        n_chan = specs.shape[-1] // 2
        real, img = specs[..., :n_chan], specs[..., n_chan:]

        l = beta.sample()

        real = l*real + (1-l)*tf.gather(real, indices, axis=0)
        img = l*img + (1-l)*tf.gather(img, indices, axis=0)
        
        mag = tf.math.sqrt(real**2 + img**2)
        phase = tf.math.atan2(img, real)

        specs = tf.concat([mag, phase], axis=-1)
        labels = tf.cast(labels, tf.float32)
        labels = l*labels + (1-l)*tf.gather(labels, indices, axis=0)
        
        return specs, labels

    return _mixup


def magphase_to_mel(num_mel_bins=80, 
                    num_spectrogram_bins=257, 
                    sample_rate=16000):
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate)

    def _magphase_to_mel(x, y=None):
        '''
        x: [batch_size, freq, time, chan2]

        output: [batch_size, mel_freq, time, chan]
        '''
        x = x[..., :tf.shape(x)[-1] // 2] # remove phase
        x = tf.tensordot(x, mel_matrix, axes=[1, 0]) # [b, time, chan, mel]
        x = tf.transpose(x, perm=[0, 3, 1, 2])

        if y is None:
            return x
        return x, y
    return _magphase_to_mel 


def log_magphase(specs, labels=None):
    n_chan = specs.shape[-1] // 2
    specs = tf.concat(
            [tf.math.log(specs[..., :n_chan]+EPSILON), specs[..., n_chan:]],
            axis=-1)
    if labels is not None:
        return specs, labels
    return specs


def minmax_norm_magphase(specs, labels=None):
    n_chan = specs.shape[-1] // 2
    mag = specs[..., :n_chan]
    phase = specs[..., n_chan:]
    axis = tuple(range(1, len(specs.shape)))

    mag_max = tf.math.reduce_max(mag, axis=axis, keepdims=True)
    mag_min = tf.math.reduce_min(mag, axis=axis, keepdims=True)
    phase_max = tf.math.reduce_max(phase, axis=axis, keepdims=True)
    phase_min = tf.math.reduce_min(phase, axis=axis, keepdims=True)

    specs = tf.concat(
        [(mag-mag_min)/(mag_max-mag_min+EPSILON),
         (phase-phase_min)/(phase_max-phase_min+EPSILON)],
        axis=-1)

    if labels is not None:
        return specs, labels
    return specs


""" COMPLEX-SPECTROGRAMS """
def complex_to_magphase(complex_tensor, y=None):
    n_chan = complex_tensor.shape[-1] // 2
    real = complex_tensor[..., :n_chan]
    img = complex_tensor[..., n_chan:]

    mag = tf.math.sqrt(real**2 + img**2)
    phase = tf.math.atan2(img, real)

    magphase = tf.concat([mag, phase], axis=-1)

    if y is None:
        return magphase
    return magphase, y


def magphase_to_complex(magphase):
    n_chan = magphase.shape[-1] // 2
    mag = magphase[..., :n_chan]
    phase = magphase[..., n_chan:]

    real = mag * tf.cos(phase)
    img = mag * tf.sin(phase)

    return tf.concat([real, img], axis=-1)


def phase_vocoder(complex_spec: tf.Tensor,
                  rate: float=1.) -> tf.Tensor:
    """
    https://pytorch.org/audio/_modules/torchaudio/functional.html#phase_vocoder

    complex_spec: [freq, time, chan*2] 
                  [..., :chan] = real, [..., chan:] = imag
    rate: float > 0.
    """
    if rate == 1:
        return complex_spec

    # shape = tf.shape(complex_spec)
    freq = complex_spec.shape[0]
    hop_length = freq - 1 # n_fft // 2
    n_chan = complex_spec.shape[-1] // 2

    def angle(spec):
        return tf.math.atan2(spec[..., n_chan:], spec[..., :n_chan])

    phase_advance = tf.linspace(
        0., np.pi * tf.cast(hop_length, 'float32'), freq)
    phase_advance = tf.reshape(phase_advance, (-1, 1, 1))
    time_steps = tf.range(
        0, tf.shape(complex_spec)[1], rate, dtype=complex_spec.dtype)

    spec = tf.pad(
        complex_spec,
        [[0, 0] if i != 1 else [0, 2] for i in range(len(complex_spec.shape))])

    spec_0 = tf.gather(spec, tf.cast(time_steps, 'int32'), axis=1)
    spec_1 = tf.gather(spec, tf.cast(time_steps+1, 'int32'), axis=1)

    angle_0 = angle(spec_0)
    angle_1 = angle(spec_1)

    norm_0 = tf.norm(
        tf.transpose(tf.reshape(spec_0, (freq, -1, 2, n_chan)), (0, 1, 3, 2)),
        2, axis=-1)
    norm_1 = tf.norm(
        tf.transpose(tf.reshape(spec_1, (freq, -1, 2, n_chan)), (0, 1, 3, 2)),
        2, axis=-1)

    # Compute Phase Accum
    phase_0 = angle(spec[..., :1, :]) # first frame angle
    phase = angle_1 - angle_0 - phase_advance
    phase = phase - 2 * np.pi * tf.math.round(phase / (2 * np.pi))
    phase = phase + phase_advance
    phase = tf.concat([phase_0, phase[:, :-1]], axis=1)
    phase_acc = tf.cumsum(phase, 1)

    alphas = tf.reshape(time_steps % 1., (1, -1, 1))
    mag = alphas * norm_1 + (1 - alphas) * norm_0

    real = mag * tf.cos(phase_acc)
    imag = mag * tf.sin(phase_acc)

    spec = tf.concat([real, imag], axis=-1)
    return spec

