from options import Options
from audio_utils import AU
import utils
import numpy as np
import librosa

if __name__ == '__main__':
    opt = Options().parse_test()
    files_path = utils.getAudioFilesPath(opt.input)
    au = AU(opt)
    
    waves = [au.load(i) for i in files_path[:10]]
    stfts = [au.wavToSTFT(i[0]) for i in waves]
    cqts = [au.wavToCQT(i[0]) for i in waves]
    [librosa.magphase(i) for i in stfts]