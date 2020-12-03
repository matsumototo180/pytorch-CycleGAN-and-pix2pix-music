# wavまたはmp3形式のオーディオファイルをCQT変換し、dbスケール変換、-1〜1の値に変換した振幅行列と位相行列を出力する
#
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

def magphaseCQT(cqt_complex, crop = False, cl = 128):
    # crop matrix
    if crop:
        cqt_complex = cqt_complex[:,:cl]

    M, P = librosa.magphase(cqt_complex)
    Mdb = librosa.amplitude_to_db(M)
    # normalize values between -1 and 1
    Mdb_normed = 2*((Mdb - Mdb.min())/(Mdb.max() - Mdb.min())) - 1
    
    return Mdb_normed, P

if __name__ == '__main__':
    # opt = Options().parse(input="/home/matsumoto/Documents/data/music/fma/fma_small_separated/Rock/")
    opt = Options().parse_cmdargs()

    bins_per_octave = 12 * 12
    n_octaves = 8
    n_bins = bins_per_octave * n_octaves
    
    print("reading audio files")
    files_path = utils.getAudioFilesPath(opt.input)
    waves = [librosa.load(i, sr=opt.sr, mono=opt.stereo) for i in files_path]
    
    print("calculating cqts")
    cqts = [librosa.cqt(i[0], opt.sr, opt.hl, n_bins=n_bins, bins_per_octave=bins_per_octave) for i in waves]
    magphases = [magphaseCQT(i, opt.crop, opt.cl) for i in cqts]
    filenames = [i.stem for i in files_path]
    out_path_amp = Path(opt.output) / Path("cqt_amp")
    out_path_phase = Path(opt.output) / Path("cqt_phase")

    print("saving npy files")
    for i, npy in enumerate(magphases):
        out_path = (out_path_amp / Path(filenames[i]))
        if len(npy[0][:,0]) % 2 != 0 :
            utils.saveNpy(npy[0][:-1,:], out_path)
        utils.saveNpy(npy[0], out_path)
    
    for i, npy in enumerate(magphases):
        out_path = (out_path_phase / Path(filenames[i]))
        if len(npy[0][:,0]) % 2 != 0 :
            utils.saveNpy(npy[1][:-1,:], out_path)
        utils.saveNpy(npy[1], out_path)