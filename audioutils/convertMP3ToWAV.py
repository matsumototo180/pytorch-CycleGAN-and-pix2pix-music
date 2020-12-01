from options import Options
from audio_utils import AU
import utils

if __name__ == '__main__':
    opt = Options().parse()
    files_path = utils.getAudioFilesPath(opt.input)
    au = AU(opt)

    mp3files_path = [i for i in files_path if i.suffix == ".mp3"]

    count = 0
    for i in mp3files_path:
        count += 1
        out = au.convertMP3ToWAV(i)
        print("processing:" + str(count) + "/" + str(len(mp3files_path)) + "   " + out)
    print("complete")