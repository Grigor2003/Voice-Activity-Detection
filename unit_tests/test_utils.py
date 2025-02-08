from other.data.datasets import OpenSLRDataset
from other.data.processing import WaveToMFCCConverter

clean_audios_path = r'../datasets/train-clean-100(converted to 8000 sr)'
clean_labels_path = r'../datasets/8000_10_50_webrtc_labels_lite.csv'
dataset = OpenSLRDataset(clean_audios_path, clean_labels_path)

mfcc_converter = WaveToMFCCConverter(
            n_mfcc=64,
            sample_rate=dataset.sample_rate,
            win_length=400,
            hop_length=200)
