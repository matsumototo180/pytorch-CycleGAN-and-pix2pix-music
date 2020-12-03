from pathlib import Path
import os
import numpy as np

def getAudioFilesPath(path):
    
    p_path = Path(path)

    if p_path.is_file():
        files_path = list([p_path])
    else:
        files_path = (list(p_path.iterdir()))
        files_path = [f for f in files_path if (f.suffix == ".wav") or (f.suffix == ".mp3")]

    return files_path

def saveNpy(npyarray, output_path):
    output_path = Path(output_path)
    if not output_path.parent.exists():
        os.makedirs(output_path.parent)
    np.save(output_path, npyarray)