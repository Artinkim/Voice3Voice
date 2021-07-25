import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as waveWrite
from PIL import Image
def audio_to_spectrogram(filename):
    y, sr = librosa.load(filename)
    window_size = 512
    hop_length = 64

    # if not os.path.isdir(output):
    #     os.mkdir(output)
    stft = librosa.feature.melspectrogram(y, n_fft=window_size, hop_length=hop_length,n_mels=3500, fmin=0.0, fmax=20000)
    #stft = np.swapaxes(stft, 0, 1)
    #stft = librosa.stft(y, n_fft=window_size, hop_length=hop_length,)
    #print(stft)
    spectrogram_to_image(stft,"thing")
    reduced = resolution_reduction(stft,stft.shape[1])
    spectrogram_to_image(reduced,"reducedThing")

    #spectrogram_to_audio(stft,"convert.wav",hop_length,sr)
    print(stft.shape,window_size,hop_length)
    #stft = stft.real

    #tft -= stft.min()

    #stft /= stft.max()
    #print(stft,stft.max(),stft.min())

    #np.save(f'{output}/{tf}.npy', stft)

    #spectrogram_to_image(stft, f'{output}/{0}')

    return stft
def resolution_reduction(arr,dilatedWindowSize):
    base = arr.shape[1]**(1/float(dilatedWindowSize))
    indicies = base ** np.arange(dilatedWindowSize)
    print(indicies,indicies.astype(np.integer))
    reduced = arr[:,indicies.astype(np.integer)]
    print(reduced,reduced.mean())
    return reduced
def spectrogram_to_audio(arr, output, window_size, sr):
    print(sr)
    audio = librosa.feature.inverse.mel_to_stft(arr, sr=sr,n_fft=512)
    norm = librosa.core.istft(audio, hop_length = 64)
    waveWrite(output, sr, norm)
    audio = librosa.core.griffinlim(audio)
    waveWrite("griffLim"+output, sr, audio)
    audio = librosa.core.griffinlim(arr)
    waveWrite("griffLimOnMel"+output, sr, audio)
def spectrogram_to_image(spec,name):
    img = spec.copy()
    avg = spec.mean()

    img -=img.min()
    img *= 255/avg
    img = np.flip(img, 0)
    Image.fromarray(img).convert('RGB').save(name+".png")
def plotSpec(stft):
    stf = np.abs(stft)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(stf,ref=np.max),y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Log')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig2, ax2 = plt.subplots()
    img2 = librosa.display.specshow(librosa.core.amplitude_to_db(stf,ref=np.max),y_axis='mel', x_axis='time', ax=ax2)
    ax2.set_title('Mel')
    fig2.colorbar(img2, ax=ax2, format="%+2.0f dB")

    fig3, ax3 = plt.subplots()
    img3 = librosa.display.specshow(librosa.core.amplitude_to_db(stf,ref=np.max),y_axis='linear', x_axis='time', ax=ax3)
    ax3.set_title('Linear')
    fig3.colorbar(img3, ax=ax3, format="%+2.0f dB")

    fig2.show()
    fig3.show()
    fig.show()
    plt.show()
stft = audio_to_spectrogram("1000_15000hz.wav")
#plotSpec(stft)
#plotSpec(stft)
