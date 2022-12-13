from EDFParser import EDFParser


def main():
    path = "/Volumes/Databases/TUH_EEG/v3.0.0/edf/train/normal/01_tcp_ar/aaaaaanr_s003_t000.edf"
    
    EDF = EDFParser()
    z = EDF.parse_file(path)
    print(z)
        
        
if __name__ == '__main__':
     main()