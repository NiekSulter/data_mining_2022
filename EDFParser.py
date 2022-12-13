from pyedflib import highlevel
import numpy as np


class EDFParser:
    def __init__(self) -> None:
        pass
        
    def parse_file(self, path):
        signals, signal_headers = self.read_file(path)
        z = self.grab_channels_of_intrest(signals, signal_headers)
        
        return z
        
    def read_file(self, path):
        signals, signal_headers, _ = highlevel.read_edf(path)
        
        return signals, signal_headers
    
    def grab_channels_of_intrest(self, signals, signal_headers):
        x, y, z = None, None, None
    
        for i, j in zip(signals, signal_headers):
            if j['label'] == 'EEG T5-REF':
                x = i
            if j['label'] == 'EEG O1-REF':
                y = i
        
        if x is not None and y is not None:
            z = np.subtract(x, y)
            
        return z