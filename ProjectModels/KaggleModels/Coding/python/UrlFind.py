import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def urlFind():
    cwdPath = str(Path.cwd()).split('\\')
    urlMainFolders = ''
    for urlFolders in cwdPath:
        if urlFolders == 'FinalProjesi':
            urlMainFolders += urlFolders + '\\'
            break
        else:
            urlMainFolders += urlFolders + '\\'
    print(urlMainFolders)
    return urlMainFolders


urlFind()
