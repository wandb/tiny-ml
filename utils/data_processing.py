import wandb
from pathlib import Path
import wave
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cmsisdsp
from numpy import pi as PI
import tensorflow as tf
import cv2

def augmenter(audio,sample_rate):
    from audiomentations import (
        Compose, AddGaussianNoise, TimeStretch, 
        PitchShift, Shift,RoomSimulator,
        ClippingDistortion,AddBackgroundNoise)
    audio = np.array([audio]).astype(np.float32)[0]
    augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.1),
    TimeStretch(min_rate=0.8, max_rate=0.9, p=0.1),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.3),
    RoomSimulator(p=0.5),
    ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=10, p = 0.1),
    AddBackgroundNoise(sounds_path='test.wav',p=0.5)
    ])
    return augment(samples=audio, sample_rate=sample_rate).astype(np.int16)[:44100]

def plot_spectrogram(spectrogram,sample_rate, vmax=None):
    '''a func to computer spect and save as a png then read as np.array'''
    plt.ioff()
    transposed_spectrogram = tf.transpose(spectrogram)
    height = transposed_spectrogram.shape[0]
    X = np.arange(transposed_spectrogram.shape[1])
    Y = np.arange(height * int(sample_rate / 256), step=int(sample_rate / 256))

    fig, ax = plt.subplots(1,1)
    ax.pcolormesh(X, Y, tf.transpose(spectrogram), vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    
    fig.tight_layout()
    plt.savefig('img.jpg',dpi=300)
    plt.close()
    return cv2.imread('img.jpg')

def segment(fid:str, chunk:int):
    # file to extract the snippet from
    data = [ ]
    with wave.open(fid, "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        # set position in wave to start of segment
        for sec in range(6):
            if sec !=0:
                sec = 4/sec
            infile.setpos(int(sec * framerate))
            data.append(infile.readframes(chunk * framerate))   
    return data, (nchannels, sampwidth , framerate)

def log_wandb_artifact(run:wandb.run,path:Path):
    if path.is_file():
        path = path.parent
    artifact = wandb.Artifact(type=path.parent.name,name=path.name)
    artifact.add_dir(str(path))
    run.log_artifact(artifact)

def read_wav(fid:str):
    # Read file to get buffer                                                                                               
    ifile = wave.open(fid)
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)
    # Convert buffer to float32 using NumPy                                                                                 
    return np.frombuffer(audio, dtype=np.int16)

class Arm_spect:
    
    def __init__(self):
        self.window_size = 512
        self.step_size = 64
        self.hanning_window_f32 = np.zeros(self.window_size)
        for i in range(self.window_size): 
            self.hanning_window_f32[i] = 0.5 * (1 - cmsisdsp.arm_cos_f32(2 * PI * i / self.window_size ))
        self.hanning_window_q15 = cmsisdsp.arm_float_to_q15(self.hanning_window_f32)
        self.rfftq15 = cmsisdsp.arm_rfft_instance_q15()
        self.status = cmsisdsp.arm_rfft_init_q15(self.rfftq15, self.window_size, 0, 1)
        
    def get_arm_spectrogram(self,waveform):
        num_frames = int(1 + (len(waveform) - self.window_size) // self.step_size)
        fft_size = int(self.window_size // 2 + 1)
        # Convert the audio to q15
        waveform_q15 = cmsisdsp.arm_float_to_q15(waveform)
        # Create empty spectrogram array
        spectrogram_q15 = np.empty((num_frames, fft_size), dtype = np.int16)
        start_index = 0
        for index in range(num_frames):
            # Take the window from the waveform.
            window = waveform_q15[start_index:start_index + self.window_size]
            # Apply the Hanning Window.
            window = cmsisdsp.arm_mult_q15(window, self.hanning_window_q15)
            # Calculate the FFT, shift by 7 according to docs
            window = cmsisdsp.arm_rfft_q15(self.rfftq15, window)
            # Take the absolute value of the FFT and add to the Spectrogram.
            spectrogram_q15[index] = cmsisdsp.arm_cmplx_mag_q15(window)[:fft_size]
            # Increase the start index of the window by the overlap amount.
            start_index += self.step_size
          # Convert to numpy output ready for keras
        return cmsisdsp.arm_q15_to_float(spectrogram_q15).reshape(num_frames,fft_size) * 512