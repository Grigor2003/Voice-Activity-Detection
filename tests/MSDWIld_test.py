from other.data.audio_utils import MSDWildDataset

msd_wavs_dir = "../datasets/MSDWild/raw_wav"
msd_labs_dir = "../datasets/MSDWild/rttm_label/few.val.rttm"

model_id = 0
target_rate = 8000

msd = MSDWildDataset(msd_wavs_dir, msd_labs_dir, target_rate)


