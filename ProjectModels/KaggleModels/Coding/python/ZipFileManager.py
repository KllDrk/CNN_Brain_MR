import zipfile
from pathlib import Path

from ProjectModels.KaggleModels.Coding.python.UrlFind import urlFind
import shutil

MainFolderUrl = str(Path.cwd().parent.parent.parent) +"\\"
print("MainFolder:" + MainFolderUrl)

x = "Coding\\Datasets"


def ExtractZipFile(targetFile, sendFile, zipName):
    print("ZipFileManager:" + f'{MainFolderUrl}{sendFile}')
    with zipfile.ZipFile(f'{targetFile}\\{zipName}', 'r') as my_zip:
        my_zip.extractall(f'{MainFolderUrl}{sendFile}')
        my_zip.close()
    pass


def CompressionZipFolder(targetFile, sendFile, zipName):
    shutil.make_archive(f'{zipName}', 'zip', f'{targetFile}{zipName}')
    shutil.copy(targetFile + zipName + '.zip', sendFile + zipName + '.zip')
    pass


def StarterExtract(targetFile, sendFile, zipName):
    ExtractZipFile(targetFile, sendFile, zipName)
    pass


def StarterCompression(targetFile, sendFile, zipName):
    CompressionZipFolder(targetFile, sendFile, zipName)
    pass
