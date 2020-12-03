# wavまたはmp3形式のオーディオファイルをSTFT変換し、振幅行列と位相行列を出力する
# 【コマンドライン引数】
# --input   入力ファイルのパスまたはディレクトリ ※必須
# --output  出力ディレクトリ（指定したパスが存在しない場合はディレクトリが作られる）
# --sr      サンプリングレート
# --stereo  この引数を指定するとステレオで処理する（未実装）
# --crop    この引数を指定するとSTFT後のデータをclでクロップする
# --wl      STFTのwindow length
# --hl      STFTのhop length
# --cl      クロップする長さ

from options import Options
import utils
import numpy as np
import librosa
from pathlib import Path

def magphaseSTFT(stft_complex, crop = False, cl = 128):
    # crop matrix
    if not crop:
        stft_complex = stft_complex[:,:cl]
    # extract magnitude spectrogram and phase spectrogram from complex matrix
    M, P = librosa.magphase(stft_complex)
    Mdb = librosa.amplitude_to_db(M)
    # normalize values between -1 and 1
    Mdb_normed = 2*((Mdb - Mdb.min())/(Mdb.max() - Mdb.min())) - 1

    return Mdb_normed, P

if __name__ == '__main__':
    # opt = Options().parse(input="/home/matsumoto/Documents/data/music/fma/fma_small_separated/Rock/")
    opt = Options().parse_cmdargs()
    
    print("reading audio files")
    files_path = utils.getAudioFilesPath(opt.input)
    waves = [librosa.load(i, sr=opt.sr, mono=opt.stereo) for i in files_path]
    
    print("calculating stfts")
    stfts = [librosa.stft(i[0], opt.wl, opt.hl, opt.wl, dtype=np.complex128) for i in waves]
    magphases = [magphaseSTFT(i, False, opt.cl) for i in stfts]
    filenames = [i.stem for i in files_path]
    out_path_amp = Path(opt.output) / Path("stft_amp")
    out_path_phase = Path(opt.output) / Path("stft_phase")

    print("saving npy files")
    for i, npy in enumerate(magphases):
        out_path = (out_path_amp / Path(filenames[i]))
        if len(npy[0][:,0]) % 2 != 0 :
            utils.saveNpy(npy[0][:-1,:], out_path)
        utils.saveNpy(npy[0], out_path)
    
    for i, npy in enumerate(magphases):
        out_path = (out_path_phase / Path(filenames[i]))
        if len(npy[0][:,0]) % 2 != 0 :
            utils.saveNpy(npy[0][:-1,:], out_path)
        utils.saveNpy(npy[1], out_path)
    