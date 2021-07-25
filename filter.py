from gammatone import gtgram
import numpy as np
from scipy.io import wavfile
from PIL import Image

class GammatoneFilterbank:
    # Initialize Gammatone filterbank
    def __init__(self,
                 sample_rate,
                 window_time,
                 hop_time,
                 num_filters,
                 cutoff_low):
        self.sample_rate = sample_rate
        self.window_time = window_time
        self.hop_time = hop_time
        self.num_filters = num_filters
        self.cutoff_low = cutoff_low
    # Make a spectrogram from a number of audio samples
    def make_spectrogram(self, audio_samples):
        return gtgram.gtgram(audio_samples,
                             self.sample_rate,
                             self.window_time,
                             self.hop_time,
                             self.num_filters,
                             self.cutoff_low)
    # Divide audio samples into dilated spectral buffers
    def make_dilated_spectral_frames(self,
                                     audio_samples,
                                     num_frames,
                                     dilation_factor):
        spectrogram = self.make_spectrogram(audio_samples)
        spectrogram = np.swapaxes(spectrogram, 0, 1)
        dilated_frames = np.zeros((len(spectrogram),
                                  num_frames,
                                  len(spectrogram[0])))

        for i in range(len(spectrogram)):
            for j in range(num_frames):
                dilation = np.power(dilation_factor, j)

                if i - dilation < 0:
                    dilated_frames[i][j] = spectrogram[0]
                else:
                    dilated_frames[i][j] = spectrogram[i - dilation]

        return dilated_frames
sr, d = wavfile.read("adinLong.wav")
a = gtgram.gtgram(d,sr,8,1,300,1000)
#a = np.rint(a)
a = a.astype(int)
print(a)
Image.fromarray(a).convert('RGB').save("spec.jpeg")
print(sr,d)
#gtgram.gtgram(d,sr,64,8,10,1000)
#s = GammatoneFilterbank(sr,512,64,30,200)
#s.make_spectrogram(d)
