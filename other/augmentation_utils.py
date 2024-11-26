import torch


def calculate_rms(tensor):
    return torch.sqrt(torch.mean(tensor ** 2))


def add_noise(audio, noise, snr_db, start, end, in_seconds=True, sample_rate=8000):
    if start < 0:
        start = 0
    if end is not None:
        if end < 0:
            end = start - end
    if in_seconds:
        start = int(start * sample_rate)
        if start >= audio.size(-1):
            return audio
        if end is None:
            end = audio.size(-1)
        else:
            end = min(audio.size(-1), int(end * sample_rate))
    if end is None:
        end = audio.size(-1)
    audio_part = audio[:, start:end]
    orig_noise = noise.clone()
    if torch.sum(torch.isnan(orig_noise)):
        print("orig noise has nan")
    while end - start > noise.size(-1):
        noise = torch.cat([noise, orig_noise], dim=1)
    noise = noise[:, : end - start]
    if torch.sum(torch.isnan(noise)):
        print("nan after repeating noise ", torch.sum(torch.isnan(orig_noise)), torch.sum(torch.isnan(noise)))
    #     noised = tf.add_noise(audio_part, noise, torch.Tensor([snr_db]))

    signal_rms = calculate_rms(audio_part)

    # Calculate RMS of the noise
    noise_rms = calculate_rms(noise)

    # Calculate desired noise RMS based on SNR
    snr = 10 ** (snr_db / 20)
    desired_noise_rms = signal_rms / snr

    # Scale noise to match the desired RMS
    scaled_noise = noise * (desired_noise_rms / (noise_rms + 1e-10))  # Adding small value to avoid division by zero

    # Add the scaled noise to the original signal
    noised = audio_part + scaled_noise

    # Debugging: Check for NaNs or Infinite
    if torch.isnan(noised).any() or torch.isinf(noised).any():
        print("Noisy waveform contains NaNs or Infinite")

    if torch.sum(torch.isnan(noised)):
        print("nan after applying noise ",
              audio[:, start:end].size(),
              noise.size(),
              torch.Tensor([snr_db]),
              torch.sum(torch.isnan(noised)))
    temp = audio.clone()
    temp[:, start:end] = noised
    if torch.sum(torch.isnan(temp)):
        print("nan after changing temp slice ", torch.sum(torch.isnan(temp)))
    return temp


def augment_sample(aw, noises=None, noise_count=1, noise_duration_range=(2, 5), snr_db=3):
    if None in [noises, noise_count, noise_duration_range, snr_db]:
        return aw.wave, None

    audio = aw.wave
    sample_rate = aw.rate
    orig_audio = audio.clone()
    augmentation_params = None

    resampled_noises = []
    for noise in noises:
        if noise.rate != sample_rate:
            temp = noise.deepcopy()
            temp.resample(sample_rate)
            resampled_noises.append(temp.wave)
        else:
            resampled_noises.append(noise.wave)
    noises = resampled_noises

    sec = audio.size(-1) / sample_rate

    if noise_count <= 0:
        return orig_audio, augmentation_params

    noises_starts, _ = torch.sort(torch.rand(noise_count) * sec)
    dur_delta = noise_duration_range[1] - noise_duration_range[0]
    noise_durations = torch.rand(noise_count) * dur_delta + noise_duration_range[0]

    noises_to_use = torch.randint(len(noises), (noise_count,))

    for i, noise_ind in enumerate(noises_to_use):
        orig_audio = add_noise(orig_audio,
                               noises[noise_ind],
                               snr_db=snr_db,
                               start=noises_starts[i],
                               end=-noise_durations[i],
                               sample_rate=sample_rate)

    augmentation_params = {"noises_starts": noises_starts,
                           "noise_durations": noise_durations,
                           "noises_to_use": noises_to_use}

    return orig_audio, augmentation_params


def generate_white_noise(count, samples_count, noise_db=1, noise_dev=0):
    dbs = torch.normal(noise_db, noise_dev, size=(count, 1))
    noise_power_linear = 10 ** (dbs / 10)
    noise = torch.randn(count, samples_count)
    noise = noise * noise_power_linear
    return noise
