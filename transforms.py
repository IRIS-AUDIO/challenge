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


def cutmix(xs, ys):
    '''
    xs : [batch, height, width, chan] np.ndarray
    ys : [batch, n_classes] np.ndarray
    '''
    xs = tf.cast(xs, tf.float32)
    ys = tf.cast(ys, tf.float32)
    batch, height, width, chan = xs.shape

    # select lambda
    sqrt_lmbda = tf.math.sqrt(tf.random.uniform([], maxval=1.))

    # set size of window and recalculate lambda
    size_h = tf.cast(height * sqrt_lmbda, dtype=tf.int32)
    size_w = tf.cast(width * sqrt_lmbda, dtype=tf.int32)

    # select window
    offset_h = tf.random.uniform([], maxval=height-size_h, dtype=tf.int32)
    offset_w = tf.random.uniform([], maxval=width-size_w, dtype=tf.int32)

    windows = tf.ones([size_h, size_w], dtype=tf.float32)
    windows = tf.pad(windows,
                     [[offset_h, height-offset_h-size_h],
                      [offset_w, width-offset_w-size_w]])
    windows = tf.reshape(windows, (1, height, width, 1))

    # shuffle
    indices = tf.math.reduce_mean(tf.ones_like(ys), axis=-1)
    indices = tf.cast(tf.math.cumsum(indices, exclusive=True), dtype='int32')
    indices = tf.random.shuffle(indices)

    # mix
    xs = xs * (1-windows) + tf.gather(xs, indices, axis=0) * windows
    mean = tf.cast((size_h * size_w) / (height * width), dtype=tf.float32)
    ys = ys * (1-mean) + tf.gather(ys, indices, axis=0) * mean

    return xs, ys


def interclass(mix_func):
    def _interclass(xs, ys):
        '''
        xs : [batch, height, width, chan] np.ndarray
        ys : [batch, n_classes] np.ndarray
        '''
        xs_, ys_ = [], []
        cls = tf.argmax(ys, axis=-1)

        for i in range(ys.shape[-1]):
            select = tf.squeeze(tf.where(cls == i), axis=-1)
            xs_temp = tf.gather(xs, select, axis=0)
            ys_temp = tf.gather(ys, select, axis=0)
            xs_temp, ys_temp = mix_func(xs_temp, ys_temp)
            xs_.append(xs_temp)
            ys_.append(ys_temp)

        return tf.concat(xs_, axis=0), tf.concat(ys_, axis=0)
    return _inter


def interbinary(mix_func):
    ''' 
    modified interclass mixup
    returns mixup(audio with voice) + mixup(non-voice)
    '''
    def _interbinary(xs, ys):
        '''
        xs : [batch, height, width, chan] np.ndarray
        ys : [batch, n_classes] np.ndarray
        '''
        xs_, ys_ = [], []
        cls = tf.argmax(ys, axis=-1)

        for i in range(2):
            if i == 0:
                select = tf.where(cls != ys.shape[-1]-1)
            else:
                select = tf.where(cls == ys.shape[-1]-1)
            select = tf.squeeze(select, axis=-1)
            xs_temp = tf.gather(xs, select, axis=0)
            ys_temp = tf.gather(ys, select, axis=0)
            xs_temp, ys_temp = mix_func(xs_temp, ys_temp)
            xs_.append(xs_temp)
            ys_.append(ys_temp)

        return tf.concat(xs_, axis=0), tf.concat(ys_, axis=0)
    return _interbinary


""" MAGNITUDE-PHASE SPECTROGRAM """
def random_magphase_flip(spec, label):
    flip = tf.cast(tf.random.uniform([]) > 0.5, spec.dtype)
    n_chan = spec.shape[-1] // 2
    chans = tf.reshape(tf.range(n_chan*2), (2, n_chan))
    chans = tf.reshape(tf.reverse(chans, axis=[-1]), (-1,))
    spec = flip*spec + (1-flip)*tf.gather(spec, chans, axis=-1)

    flip = tf.cast(flip, label.dtype)
    label = flip*label \
            + (1-flip)*tf.concat(
                [tf.reverse(label[..., :-1], axis=(-1,)), label[..., -1:]],
                axis=-1)

    return spec, label


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


def complex_to_magphase(complex_tensor):
    n_chan = complex_tensor.shape[-1] // 2
    real = complex_tensor[..., :n_chan]
    img = complex_tensor[..., n_chan:]

    mag = tf.math.sqrt(real**2 + img**2)
    phase = tf.math.atan2(img, real)

    return tf.concat([mag, phase], axis=-1)


def magphase_to_complex(magphase):
    n_chan = magphase.shape[-1] // 2
    mag = magphase[..., :n_chan]
    phase = magphase[..., n_chan:]

    real = mag * tf.cos(phase)
    img = mag * tf.sin(phase)

    return tf.concat([real, img], axis=-1)


def merge_complex_specs(background, 
                        voice_and_label, 
                        noise=None,
                        n_frame=300, 
                        time_axis=1,
                        prob=0.9,
                        noise_prob=0.3,
                        min_voice_ratio=2/3,
                        min_noise_ratio=1/2,
                        n_voice_classes=10):
    '''
    INPUT
    background: [freq, time, chan2] 
    voice_and_label: tuple ([freq, time, chan2], int)
    noise: None or [freq, time, chan2]
    n_frame: number of output frames
    time_axis: time axis, default=1
    prob: probability of adding voice
    noise_prob: prob of adding noise
    min_voice_ratio: minimum ratio of voice overlap with background
    min_noise_ratio: minimum ratio of noise overlap with background
    n_voice_classes: 

    OUTPUT
    complex_spec: [freq, time, chan*2]
    output_label: one_hot [n_voice_classes + 1]
    '''
    voice, label = voice_and_label
    output_shape = tuple(
        [s if i != time_axis else n_frame
         for i, s in enumerate(background.shape)])
    n_dims = len(output_shape)

    # background
    bg_frame = tf.shape(background)[time_axis]
    background = tf.tile(
        background, 
        [1 if i != time_axis else (n_frame+bg_frame-1) // bg_frame 
         for i in range(n_dims)])
    complex_spec = tf.image.random_crop(background, output_shape)

    # voice
    v_bool = tf.random.uniform([]) < prob
    if v_bool: # OVERLAP
        v_ratio = tf.math.pow(10., -tf.random.uniform([], maxval=2)) # SNR0~-20
        v_frame = tf.cast(tf.shape(voice)[time_axis], tf.float32)
        if v_frame < n_frame:
            voice = tf.pad(
                voice,
                [[0, 0] 
                 if i != time_axis 
                 else [n_frame - tf.cast(min_voice_ratio*v_frame, tf.int32)]*2
                 for i in range(n_dims)])
        voice = tf.image.random_crop(voice, output_shape)
        complex_spec += v_ratio * voice
    else:
        label = n_voice_classes # non-voice audio
    
    # noise
    if noise is not None:
        if tf.random.uniform([]) < noise_prob:
            v_ratio = tf.math.pow(10., -tf.random.uniform([], maxval=2)) # SNR0~-20
            v_frame = tf.cast(tf.shape(voice)[time_axis], tf.float32)
            if v_frame < n_frame:
                voice = tf.pad(
                    voice,
                    [[0, 0] 
                     if i != time_axis 
                     else [n_frame - tf.cast(min_noise_ratio*v_frame, tf.int32)]*2
                     for i in range(n_dims)])
            voice = tf.image.random_crop(voice, output_shape)
            complex_spec += v_ratio * voice

    output_label = tf.one_hot(label, n_voice_classes+1)
    return complex_spec, output_label


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
    time_steps = tf.range(0, tf.shape(complex_spec)[1], rate, dtype=complex_spec.dtype)

    spec = tf.pad(complex_spec,
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

