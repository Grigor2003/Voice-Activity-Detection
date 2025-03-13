import torch

from other.data.audio_utils import AudioWorker


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


def augment_with_noises(aw: AudioWorker, noises=None, noise_count=1, noise_duration_range=(2, 5), snr_db=3, **kwargs):
    if None in [noises, noise_count, noise_duration_range, snr_db]:
        return {"noise_None": True}

    for noise in noises:
        noise.resample(aw.rate)

    if noise_count <= 0:
        return {"noise_count": noise_count}

    noises_starts, _ = torch.sort(torch.rand(noise_count) * aw.duration_s)
    dur_delta = noise_duration_range[1] - noise_duration_range[0]
    noise_durations = torch.rand(noise_count) * dur_delta + noise_duration_range[0]

    noises_to_use = torch.randint(len(noises), (noise_count,))

    for i, noise_ind in enumerate(noises_to_use):
        aw.wave = add_noise(aw.wave, sample_rate=aw.rate, noise=noises[noise_ind].wave,
                            snr_db=snr_db, start=noises_starts[i], end=-noise_durations[i])

    return {"noises_starts": noises_starts,
            "noise_durations": noise_durations,
            "noises_to_use": noises_to_use}


def augment_volume_gain(aw, gain_function='random', effect_ratio=(0.1, 0.7), value_range=(0.15, 1), inverse=0.5):
    functions = ['sin', 'woods']
    if gain_function == "random":
        ind = torch.randint(0, len(functions), (1,)).item()
        gain_function = functions[ind]
    if not isinstance(effect_ratio, (float, int)):
        i_min, i_max = effect_ratio
        mean, std = 0.5, 0.25
        r = torch.clamp(mean + torch.randn(1) * std, 0.0, 1.0).item()
        effect_ratio = i_min + r * (i_max - i_min)

    match gain_function:
        case 'sin':
            gain = torch.sin(torch.linspace(0, 2 * torch.pi, steps=aw.length)) ** (2 * int(25 ** effect_ratio))
        case 'woods':
            a, b, c = effect_ratio, effect_ratio, 0.5
            n = lambda x: 1 / torch.tan(torch.pi * (2 * torch.abs(x - c) / b))
            v = lambda x: 1 / (1 + torch.exp(n(x)) ** (100 ** a))
            fr, to = c - b / 2, c + b / 2
            gain = torch.ones(aw.length)
            s_fr, s_to = int(aw.length * fr), int(aw.length * to)
            gain[s_fr: s_to] = v(torch.linspace(fr, to, steps=s_to - s_fr))
        case _:
            ValueError("Unsupported gain function. Choose 'sin' or 'woods'.")
            return

    was_inversed = torch.rand(1).item() < inverse
    if was_inversed:
        gain = 1 - gain

    min_gain, max_gain = value_range
    gain = min_gain + gain * (max_gain - min_gain)
    aw.wave *= torch.abs(gain.to(aw.wave.device))

    return {"was_inversed": was_inversed,
            "gain_function": gain_function,
            "effect_ratio": effect_ratio}


def generate_white_noise(count, samples_count, noise_db=1, noise_dev=0):
    dbs = torch.normal(noise_db, noise_dev, size=(count, 1))
    noise_power_linear = 10 ** (dbs / 20)
    noise = torch.randn(count, samples_count)
    noise = noise * noise_power_linear
    return noise
