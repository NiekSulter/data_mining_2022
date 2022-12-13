from EDFParser import EDFParser
import glob


def single_file():
    path = "/Volumes/Databases/TUH_EEG/v3.0.0/edf/train/normal/01_tcp_ar/aaaaaanr_s003_t000.edf"
    
    EDF = EDFParser()
    z = EDF.parse_file(path)
    print(z)


def multi_file():
    path_list = [path for path in glob.iglob('/Volumes/Databases/TUH_EEG/v3.0.0/edf/train/normal/01_tcp_ar/*.edf')]
    
    EDF = EDFParser()
    for path in path_list:
        z = EDF.parse_file(path)
        print(z)