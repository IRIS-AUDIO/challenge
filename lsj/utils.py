import os, pickle
import numpy as np
import tensorflow as tf
from functools import partial
from pipeline import *
from transforms import *
EPSILON = 1e-8

def to_class_labels(x, y):
    '''
    INPUT - y : [n_voices, n_frames, 30]
    OUTPUT - y: [3, 10]
    '''
    tf.print(y.shape)
    y = tf.reduce_max(y, axis=1)
    y = tf.reduce_sum(y, axis=0)
    y = tf.reshape(y, [3, 10])
    return x, y


''' 
UTILS FOR FRAMES AND WINDOWS 
'''
def seq_to_windows(seq, 
                   window, 
                   skip=1,
                   padding=True, 
                   **kwargs):
    '''
    INPUT:
        seq: np.ndarray
        window: array of indices
            ex) [-3, -1, 0, 1, 3]
        skip: int
        padding: bool
        **kwargs: params for np.pad

    OUTPUT:
        windows: [n_windows, window_size, ...]
    '''
    window = np.array(window - np.min(window)).astype(np.int32)
    win_size = max(window) + 1
    windows = window[np.newaxis, :] \
            + np.arange(0, len(seq), skip)[:, np.newaxis]
    if padding:
        seq = np.pad(
            seq,
            [[win_size//2, (win_size-1)//2]] + [[0, 0]]*len(seq.shape[1:]),
            **kwargs)

    return np.take(seq, windows, axis=0)


def windows_to_seq(windows,
                   window,
                   skip=1):
    '''
    INPUT:
        windows: np.ndarray (n_windows, window_size, ...)
        window: array of indices
        skip: int

    OUTPUT:
        seq
    '''
    n_window = windows.shape[0]
    window = np.array(window - np.min(window)).astype(np.int32)
    win_size = max(window)

    seq_len = (n_window-1)*skip + 1
    seq = np.zeros([seq_len, *windows.shape[2:]], dtype=windows.dtype)
    count = np.zeros(seq_len)

    for i, w in enumerate(window):
        indices = np.arange(n_window)*skip - win_size//2 + w
        select = np.logical_and(0 <= indices, indices < seq_len)
        seq[indices[select]] += windows[select, i]
        count[indices[select]] += 1
    
    seq = seq / (count + EPSILON)
    return seq


'''
DATASET
'''
def list_to_generator(dataset: list):
    def _gen():
        if isinstance(dataset, tuple):
            for z in zip(*dataset):
                yield z
        else:
            for data in dataset:
                yield data
    return _gen


'''
MODEL
'''
def apply_kernel_regularizer(model, kernel_regularizer):
    model = tf.keras.models.clone_model(model)
    layer_types = (tf.keras.layers.Dense, tf.keras.layers.Conv2D)
    for layer in model.layers:
        if isinstance(layer, layer_types):
            layer.kernel_regularizer = kernel_regularizer

    model = tf.keras.models.clone_model(model)
    return model

def make_dataset(config, training=True):
        # Load required datasets
        if training:
            backgrounds = load_data(config.background_sounds)
            voices = load_data(config.voices)
            labels = load_data(config.labels)
        else:
            backgrounds = load_data(config.test_background_sounds)
            voices = load_data(config.test_voices)
            labels = load_data(config.test_labels)
        labels = np.eye(config.n_classes, dtype='float32')[labels] # to one-hot vectors
        noises = load_data(config.noises)

        # Make pipeline and process the pipeline
        pipeline = make_pipeline(backgrounds, 
                                voices, labels,
                                noises,
                                n_frame=config.n_frame,
                                max_voices=config.max_voices,
                                max_noises=config.max_noises,
                                n_classes=config.n_classes,
                                snr=config.snr)
        if config.predict == 'sample':
            pipeline = pipeline.map(to_class_labels)
        elif config.predict == 'frame':
            pipeline = pipeline.map(to_frame_labels)
        if training:
            pipeline = pipeline.map(augment)
        pipeline = pipeline.batch(config.batch_size, drop_remainder=False)
        pipeline = pipeline.map(complex_to_magphase)
        pipeline = pipeline.map(magphase_to_mel(config.n_mels))
        pipeline = pipeline.map(minmax_log_on_mel)
        pipeline = pipeline.map(preprocess_labels)
        return pipeline.prefetch(AUTOTUNE)

def preprocess_labels(x, y):
    # preprocess y
    for i in range(5):
        y = tf.nn.max_pool1d(y, 2, strides=2, padding='SAME')
    return x, y

def minmax_log_on_mel(mel, labels=None):
    axis = tuple(range(1, len(mel.shape)))

    # MIN-MAX
    mel_max = tf.math.reduce_max(mel, axis=axis, keepdims=True)
    mel_min = tf.math.reduce_min(mel, axis=axis, keepdims=True)
    mel = (mel-mel_min) / (mel_max-mel_min+EPSILON)

    # LOG
    mel = tf.math.log(mel + EPSILON)

    if labels is not None:
        return mel, labels
    return mel

def make_pipeline(backgrounds, # a list of backgrounds noises
                  voices, # a list of human voicess
                  labels, # a list of labelss of human voicess
                  noises=None, # a list of additional noises
                  n_frame=300, # number of frames per sample
                  max_voices=10,
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
    v_dataset = v_dataset.padded_batch(
        max_voices, padded_shapes=([freq, None, chan], [n_classes]))

    # NOISES
    if noises is not None:
        n_dataset = tf.data.Dataset.from_generator(
            list_to_generator(noises),
            tf.float32,
            tf.TensorShape([freq, None, chan]))
        n_dataset = n_dataset.repeat().shuffle(len(noises))
        n_dataset = n_dataset.padded_batch(
            max_noises, padded_shapes=[freq, None, chan])
        dataset = tf.data.Dataset.zip((b_dataset, v_dataset, n_dataset))
    else:
        dataset = tf.data.Dataset.zip((b_dataset, v_dataset))

    dataset = dataset.map(partial(merge_complex_specs,
                                  n_frame=n_frame,
                                  n_classes=n_classes,
                                  **kwargs))
    return dataset

def augment(specs, labels, time_axis=1, freq_axis=0):
    specs = mask(specs, axis=time_axis, max_mask_size=24, n_mask=8) # time
    specs = mask(specs, axis=freq_axis, max_mask_size=32) # freq
    specs = random_shift(specs, axis=freq_axis, width=8)
    return specs, labels

def load_data(path):
    if path.endswith('.pickle'):
        return pickle.load(open(path, 'rb'))
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError('invalid file format')