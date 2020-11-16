import pickle, torchaudio, torch
from glob import glob

def getRawData(config):
    data_path = sorted(glob(config.path + '/*.wav'))

    def load(path):
        torchaudio.load(data_path)
        return 

    data = list(map(load, data_path))

    return data_path

def _load_wav(wav_fname: str):
    '''
    OUTPUT
    complex_specs: list of complex spectrograms
                   each complex spectrogram has shape of
                   [freq, time, chan*2]
    '''
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    stft = torchaudio.transforms.Spectrogram(512, power=None).to(device)
    
    wav, r = torchaudio.load(wav_fname)
    wav = torchaudio.compliance.kaldi.resample_waveform(
        wav, r, 16000)
    wav = normalize(wav)
    wav = stft(wav).cpu()
    
    # [chan, freq, time, 2] -> [freq, time, chan, 2]
    wav = wav.numpy().transpose(1, 2, 3, 0)
    wav = wav.reshape((*wav.shape[:2], -1))

    return wav



def load_wav(config):
    wavs = sorted(glob(config.path + '/*.wav'))
    import tensorflow as tf
    from transforms import magphase_to_mel, complex_to_magphase, minmax_norm_magphase
    to_mel = magphase_to_mel(config.n_mels)
    
    wavs = list(map(_load_wav, wavs)) 
    target = max([tuple(wav.shape) for wav in wavs]) 
    wavs = list(map(lambda x: tf.pad(x, [[0, 0], [0, target[1]-x.shape[1]], [0, 0]]), 
                    wavs)) 
    wavs = tf.convert_to_tensor(wavs) 
    wavs = complex_to_magphase(wavs)
    wavs = magphase_to_mel(config.n_mels)(wavs)
    def safe_div(x, y, eps=1e-6):
        # returns safe x / max(y, epsilon)
        return x / tf.maximum(y, eps)

    def minmax_log_on_mel(mel, labels=None):
        axis = tuple(range(1, len(mel.shape)))

        # MIN-MAX
        mel_max = tf.math.reduce_max(mel, axis=axis, keepdims=True)
        mel_min = tf.math.reduce_min(mel, axis=axis, keepdims=True)
        mel = safe_div(mel-mel_min, mel_max-mel_min)

        # LOG
        mel = tf.math.log(mel + 1e-6)

        if labels is not None:
            return mel, labels
        return mel
    wavs = minmax_log_on_mel(wavs)
    return wavs


def normalize(wav):
    rms = torch.sqrt(torch.mean(torch.pow(wav, 2))) * 10
    return wav / rms
