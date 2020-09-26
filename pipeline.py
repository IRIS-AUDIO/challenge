import tensorflow as tf
from functools import partial
from utils import list_to_generator


def merge_complex_specs(background, 
                        voices_and_labels, 
                        noises=None,
                        n_frame=300, 
                        n_classes=30,
                        t_axis=1, # time-axis
                        prob=0.9,
                        noise_prob=0.3,
                        min_ratio=2/3,
                        min_noise_ratio=1/2):
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
    label = tf.zeros((n_frame, n_classes), dtype='float32')

    # voices
    for v in range(tf.shape(voices)[0]):
        voice = voices[v]
        l = labels[v:v+1] # shape=[1, n_classes]

        if tf.random.uniform([]) < prob:
            # SNR 0 ~ -20
            v_ratio = tf.math.pow(10., -tf.random.uniform([], maxval=2))
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

            complex_spec += v_ratio * voice
            label += l
    
    # noise
    if noises is not None:
        for n in range(tf.shape(noises)[0]):
            noise = noises[n]
            if tf.random.uniform([]) < noise_prob:
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

    label = tf.clip_by_value(label, 0, 1)
    return complex_spec, label


def make_pipeline(backgrounds, # a list of backgrounds noises
                  voices, # a list of human voicess
                  labels, # a list of labelss of human voicess
                  noises=None, # a list of additional noises
                  batch_size=64,
                  n_frame=300, # number of frames per sample
                  max_voicess=10,
                  max_noises=10,
                  n_classes=30,
                  **kwargs):
    '''
    OUTPUT
        dataset: tf.data.Dataset
                 it only returns a raw complex spectrogram
                 and its labels
                 you have to apply augmentations (ex. mixup)
                 or preprocessing functions (ex. applying log)
                 you don't have to apply shuffle

                 complex spectrogram: [freq_bins, n_frame, chan*2]
                     [..., :chan] = real
                     [..., chan:] = imag
                 labels: [n_frame, n_classes]
    '''
    assert len(backgrounds[0].shape) == 3, 'each spec must be a 3D-tensor'
    assert len(voices) == len(labels)
    assert len(labels[0].shape) == 1 and labels[0].shape[0] == n_classes, \
           'labels must be in the form of [n_samples, n_classes]'

    # BACKGROUND NOISE (DRONE)
    freq, _, chan = backgrounds[0].shape
    b_dataset = tf.data.Dataset.from_generator(
        list_to_generator(backgrounds),
        tf.float32,
        tf.TensorShape([freq, None, chan]))
    b_dataset = b_dataset.repeat().shuffle(len(backgrounds))

    # HUMAN VOICE
    v_dataset = tf.data.Dataset.from_generator(
        list_to_generator((voices, labels)),
        (tf.float32, tf.float32),
        (tf.TensorShape([freq, None, chan]), tf.TensorShape([n_classes])))
    v_dataset = v_dataset.repeat().shuffle(len(voices))
    v_dataset = v_dataset.padded_batch(max_voicess)

    # NOISES
    if noises is not None:
        n_dataset = tf.data.Dataset.from_generator(
            list_to_generator(noises),
            tf.float32,
            tf.TensorShape([freq, None, chan]))
        n_dataset = n_dataset.repeat().shuffle(len(noises))
        n_dataset = n_dataset.padded_batch(max_noises)
        dataset = tf.data.Dataset.zip((b_dataset, v_dataset, n_dataset))
    else:
        dataset = tf.data.Dataset.zip((b_dataset, v_dataset))

    dataset = dataset.map(partial(merge_complex_specs,
                                  n_frame=n_frame,
                                  n_classes=n_classes,
                                  **kwargs))
    return dataset

