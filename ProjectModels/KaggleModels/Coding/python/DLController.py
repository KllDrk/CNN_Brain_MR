import os

from ProjectModels.KaggleModels.Coding.python.DatasetManager import DatasetManagerStarter
from ProjectModels.KaggleModels.Coding.DL.AlzheimerDL import AlzheimerDLStarter
from ProjectModels.KaggleModels.Coding.DL.TumorDL import TumorDLStarter
from ProjectModels.KaggleModels.Coding.DL.HowClassDL import HowClassDLStarter
from ProjectModels.KaggleModels.Coding.python.UrlFind import urlFind

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mainUrlFolder = urlFind()


def runningFunc():
    DatasetManagerStarter()
    HowClassDLStarter()
    AlzheimerDLStarter()
    TumorDLStarter()
    pass


def runningFuncStarter():
    runningFunc()
    pass
