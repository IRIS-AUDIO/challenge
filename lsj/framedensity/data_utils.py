import torch
import torchaudio
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
            

if __name__ == '__main__':
    import glob
    wavs = glob.glob('/codes/2020_track3/t3_audio/*.wav')
    print(wavs)
    stfts = [load_wav(wav) for wav in wavs]

    for stft in stfts:
        print(stft.shape)

