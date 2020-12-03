# cqtによって得られた振幅行列と位相行列のnpyファイルから逆cqtを行ってwavファイルを復元する
#
# 【コマンドライン引数】
# --input   振幅行列ファイルのパスまたはディレクトリ ※必須
# --phase_input   位相行列ファイルのパスまたはディレクトリ ※必須
# --output  出力ディレクトリ（指定したパスが存在しない場合はディレクトリが作られる）
# --sr      サンプリングレート
# --wl      iSTFTのwindow length
# --hl      iSTFTのhop length

from options import Options
import utils
import numpy as np
import librosa
from pathlib import Path

def generateWavFromCQT(amp, phase, sr, hl):
    
    Mdb_inormed = np.interp(amp, (amp.min(), amp.max()), (-15, 65))   
    iM = librosa.db_to_amplitude(Mdb_inormed)

    bins_per_octave = 12 * 12
    D = iM * np.exp(1j*phase)
    iy = librosa.icqt(C=D, sr=sr, hop_length=hl, bins_per_octave=bins_per_octave)
    iy = librosa.util.normalize(iy)

    return iy

if __name__ == '__main__':
    # opt = Options(inverse=True).parse(input="/home/matsumoto/Documents/data/music/processed/rock/stft_amp/", phase_input="/home/matsumoto/Documents/data/music/processed/rock/stft_phase/", output="/home/matsumoto/Documents/data/music/processed/rock/")
    opt = Options(inverse=True).parse_cmdargs()
    
    files_path_amp = utils.getFilesPath(Path(opt.input), ".npy")
    files_path_phase = utils.getFilesPath(Path(opt.phase_input), ".npy")
    
    total_length = str(min(len(files_path_amp), len(files_path_phase)))
    for i, amp_path in enumerate(files_path_amp):
        ampspec = np.load(amp_path)
        phspec = np.load(files_path_phase[i])

        ampspec = ampspec[0,:,:] if ampspec.ndim == 3 else ampspec
        ampspec = ampspec[:,:min(ampspec.shape[1], phspec.shape[1])]
        ampspec = np.append(ampspec, np.zeros([1, ampspec.shape[1]], dtype=np.float64), axis=0)
        phspec = np.append(phspec, np.zeros([1, phspec.shape[1]], dtype=np.complex128), axis=0)

        wav = generateWavFromCQT(ampspec, phspec, opt.sr, opt.hl)

        output_path = Path(opt.output) / Path("cqt_inv") / amp_path.name
        print(str(output_path.name), " [", str(i + 1), "/", total_length, "]", sep="")
        utils.saveWav(wav, output_path, opt.sr)