import torchaudio

wave, sr = torchaudio.load(r"C:\Users\narek\EpicDocuments\PythonProjects\Voice-Activity-Detection\accent-dataset\converted_wavs\19-198-0003.wav")
print(sr)