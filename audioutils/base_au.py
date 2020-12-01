import librosa
import sys
import numpy as np
import soundfile
import argparse
from pathlib import Path
from PIL import Image

class AU():
    def __init__(self, opt):
        self.opt = opt
    def load(self, path):
        sr = self.opt.sr
        mono = not self.opt.stereo
        y, sr = librosa.load(path, sr=sr, mono=mono)
        return y, sr    
    def wavToSTFT(self, wave):
        window_length = self.opt.wl
        hop_length = self.opt.hl
        # compute stft
        C = librosa.stft(wave, window_length, hop_length, window_length, dtype=np.complex128)
        return C
    def wavToCQT(self, wave, n_octaves = 8, bins_per_octave = 144):
        sr = self.opt.sr
        hop_length = self.opt.hl
        n_bins = bins_per_octave * n_octaves
        C = librosa.cqt(wave, sr, hop_length, None, n_bins, bins_per_octave)
        return C



# sr = 22050
# mono = True

# # p_path = Path("/home/matsumoto/Downloads/fma/fma_small_separated/Rock/Rock.000182.wav")
# p_path = Path("/home/matsumoto/Downloads/fma/fma_small_separated/Rock/")

# if p_path.is_file():
#     files_path = list([p_path])
# else:
#     files_path = (list(p_path.iterdir()))
#     files_path = [f for f in files_path if (".wav" in f._str) or (".mp3" in f._str)]



# y = librosa.load(p_path, sr=sr, mono=mono) 