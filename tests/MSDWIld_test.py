from other.audio_utils import MSDWildDataset

msd_wavs_dir = "../data/MSDWild/raw_wav"
msd_labs_dir = "../data/MSDWild/rttm_label/few.val.rttm"

model_id = 0
target_rate = 8000

msd = MSDWildDataset(msd_wavs_dir, msd_labs_dir, target_rate)


