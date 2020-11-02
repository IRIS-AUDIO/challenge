import tensorflow as tf
from functools import partial


def merge_complex_specs(background, 
                        voices_and_labels, 
                        noises=None,
                        n_frame=300, 
                        n_classes=30,
                        t_axis=1, # time-axis
                        min_ratio=2/3,
                        min_noise_ratio=1/2,
                        snr=-20):
    '''
    OUTPUT:
        complex_spec: (freq, time, chan2)
        labels: (n_voices, time, n_classes)
    '''
    voices, labels = voices_and_labels
    output_shape = tuple(
        [s if i != t_axis else n_frame
         for i, s in enumerate(background.shape)])
    n_dims = len(output_shape)
    axis = tuple(i for i in range(n_dims) if i != t_axis)

    # background and its label
    bg_frame = tf.shape(background)[t_axis]
    background = tf.tile(
        background, 
        [1 if i != t_axis else (n_frame+bg_frame-1) // bg_frame 
         for i in range(n_dims)])
    complex_spec = tf.image.random_crop(background, output_shape)

    # voices
    max_voices = tf.shape(voices)[0]
    n_voices = tf.random.uniform([], minval=1, maxval=max_voices, dtype='int32')
    label = tf.zeros(shape=[max_voices, n_frame, n_classes], dtype='float32')
    for v in range(n_voices):
        voice = voices[v]
        l = labels[v:v+1] # shape=[1, n_classes]

        v_ratio = tf.math.pow(10., -tf.random.uniform([], maxval=-snr/10))
        v_frame = tf.shape(voice)[t_axis]

        l = tf.tile(l, [v_frame, 1])
        mask = tf.cast(tf.reduce_max(voice, axis=axis) > 0, tf.float32)
        l *= tf.expand_dims(mask, axis=-1)

        v_frame = tf.cast(v_frame, tf.float32)
        pad_size = n_frame - tf.cast(min_ratio*v_frame, tf.int32)

        if pad_size > 0:
            voice = tf.pad(
                voice,
                [[0, 0] if i != t_axis else [pad_size] * 2
                 for i in range(n_dims)])
            l = tf.pad(l, [[pad_size]*2, [0, 0]])

        maxval = tf.shape(voice)[t_axis] - n_frame
        offset = tf.random.uniform([], maxval=maxval, dtype=tf.int32)
        voice = tf.slice(
            voice, 
            [0 if i != t_axis else offset for i in range(n_dims)],
            output_shape)
        l = tf.slice(l, [offset, 0], [n_frame, n_classes])
        l = tf.reshape(tf.one_hot(v, max_voices, dtype='float32'), (-1, 1, 1)) \
            * tf.expand_dims(l, axis=0)

        complex_spec += v_ratio * voice
        label += l
    
    # noise
    if noises is not None:
        n_noises = tf.random.uniform([], maxval=tf.shape(noises)[0], dtype='int32')
        for n in range(n_noises):
            noise = noises[n]

            # SNR 0 ~ -20
            n_ratio = tf.math.pow(10., -tf.random.uniform([], maxval=2)) 
            ns_frame = tf.cast(tf.shape(noise)[t_axis], tf.float32)
            pad_size = n_frame - tf.cast(min_noise_ratio*ns_frame, tf.int32)

            if pad_size > 0:
                noise = tf.pad(
                    noise,
                    [[0, 0] if i != t_axis else [pad_size]*2
                     for i in range(n_dims)])
            noise = tf.image.random_crop(noise, output_shape)
            complex_spec += n_ratio * noise

    return complex_spec, label


def to_frame_labels(x, y):
    """

    :param x:
    :param y: [n_voices, n_frames, n_classes]
    :return: [n_frames, n_classes]
    """
    y = tf.reduce_sum(y, axis=0)
    y = tf.clip_by_value(y, 0, 1)
    return x, y





