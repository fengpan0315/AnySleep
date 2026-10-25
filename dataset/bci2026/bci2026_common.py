"""
Shared BCI 2026 code: channel/stage constants, EDF and label-TXT loading, and
the preprocessing pipeline used by both prepare_bci2026.py (training data
creation) and predict_bci2026.py (batch inference).

The pipeline replicates dslaborg/Sleep-Datasets (preprocess.py with the
standard config/dataset settings) for this dataset, and matches
examples/predict_edf_file_logits_plain.py:

    load PSG via pyedflib (physical calibration from the EDF header)
    -> re-reference every channel against M1: EEG-M1, EOG-R-M1, EOG-L-M1
    -> trim to whole EPOCH_DURATION epochs from the EDF start
    -> bandpass FIR between FILTER_L_FREQ and FILTER_H_FREQ at the original
       sample rate, before resampling               (psg_utils filtering)
    -> resample_poly to SAMPLE_RATE                 (psg_utils psg_sampling)
    -> clip to +/- MIN_MAX_TIMES_GLOBAL_IQR x global IQR per channel
                                                    (quality_control_funcs)
    -> per-channel RobustScaler fit on the whole recording (MultiChannelScaler)
    -> float32

Label TXT layout: tab-separated '时间 [hh:mm:ss] / 睡眠期 / 持续时间' with
EPOCH_DURATION s rows, stages W/N1/N2/N3/REM, first row aligned with the EDF
start time.
"""
import logging
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pyedflib
from mne.filter import filter_data
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# input channels after re-referencing against M1, EEG group first then EOG;
# the U-Sleep train sampler draws one random EEG ('EEG') and one random EOG
# ('EOG-R'/'EOG-L') channel per window (base/data/usleep_train_dataset.py),
# the AnySleep train sampler a random subset of the combined channel pool
# (base/data/anysleep_train_dataset.py). ECG is not used.
EEG_CHANNELS = ["EEG"]
EOG_CHANNELS = ["EOG-R", "EOG-L"]
# raw reference channel subtracted from every input channel at load time;
# M1 itself is not a model input
REFERENCE_CHANNEL = "M1"
SAMPLE_RATE = 128  # output sample rate of the signals, in Hz
EPOCH_DURATION = 30  # sleep epoch duration, in seconds
# bandpass filter cutoffs (Hz), applied at the original sample rate before
# resampling, as the psg_utils _load pipeline does (filtering.apply_filtering)
FILTER_L_FREQ = 0.5
FILTER_H_FREQ = 16
# stage encoding of Sleep-Datasets (general.stages / general.rstages)
STAGE_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
STAGE_NAMES = list(STAGE_MAP)  # stage index -> name (STAGE_MAP is in index order)
MIN_MAX_TIMES_GLOBAL_IQR = 20  # quality control threshold, see preprocess.yaml


def load_psg(edf_path):
    """
    Loads the EEG/EOG channels of an EDF file (physical values, calibrated
    from the EDF header digital/physical min-max) and re-references them
    against M1: the returned columns are EEG-M1, EOG-R-M1 and EOG-L-M1.

    Returns:
        psg:    float64 ndarray of shape [N, len(EEG_CHANNELS + EOG_CHANNELS)]
        sfreq:  sample rate of the file, in Hz
        start:  naive datetime of the recording start, from the EDF header
    """
    channels = EEG_CHANNELS + EOG_CHANNELS
    read_channels = channels + [REFERENCE_CHANNEL]
    f = pyedflib.EdfReader(str(edf_path))
    try:
        labels = [f.getLabel(i) for i in range(f.signals_in_file)]
        duplicates = sorted(chan for chan in set(labels) if labels.count(chan) > 1)
        if duplicates:
            raise ValueError(
                f"{edf_path}: duplicate channel labels {duplicates} ({labels})"
            )
        missing = [chan for chan in read_channels if chan not in labels]
        if missing:
            raise ValueError(
                f"{edf_path}: channels {missing} not found in file ({labels})"
            )
        # load the channels in the requested order, independent of the file order
        label_to_ind = {chan: i for i, chan in enumerate(labels)}
        sfreqs = {
            round(f.getSampleFrequency(label_to_ind[chan]), 10)
            for chan in read_channels
        }
        if len(sfreqs) != 1:
            raise ValueError(
                f"{edf_path}: selected channels have mixed/unknown sample rates: "
                f"{sfreqs}"
            )
        sfreq = int(sfreqs.pop())
        raw = {
            chan: f.readSignal(label_to_ind[chan]) / 1.0e6 for chan in read_channels
        }
        # re-reference every input channel against M1
        ref = raw[REFERENCE_CHANNEL]
        psg = np.stack([raw[chan] - ref for chan in channels], axis=1)
        start = f.getStartdatetime()
    finally:
        f.close()
    if start is None or start.year < 1980:
        raise ValueError(f"{edf_path}: EDF header has no valid start date/time")
    return np.ascontiguousarray(psg, dtype=np.float64), sfreq, start


def load_hypnogram(txt_path):
    """
    Parses a label TXT into per-epoch sleep stage integers.

    Expects tab-separated 'hh:mm:ss / stage / duration' rows of 30 s,
    contiguous from the first row (times wrap around midnight) and aligned
    with the EDF start time. A header row before the first 'hh:mm:ss' time
    is skipped.

    Returns:
        first_time: datetime of the first row
        stages:     int64 ndarray of shape [n_epochs]
    """
    stages = []
    first_time = None
    expected_time = None  # next row's expected clock time
    with open(txt_path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"{txt_path}:{line_no}: expected 3 tab-separated fields, "
                    f"got {line.rstrip()!r}"
                )
            time_str, stage_str, dur_str = (p.strip() for p in parts)
            if first_time is None and not re.fullmatch(r"\d{2}:\d{2}:\d{2}", time_str):
                logger.debug(f"Skipping header row in {os.path.basename(txt_path)}")
                continue
            try:
                duration = int(dur_str)
            except ValueError:
                raise ValueError(
                    f"{txt_path}:{line_no}: non-integer duration {dur_str!r}"
                )
            time_ = datetime.strptime(time_str, "%H:%M:%S")
            time_sec = timedelta(hours=time_.hour, minutes=time_.minute, seconds=time_.second)
            if first_time is None:
                first_time = time_
                expected_time = time_sec
            elif time_sec != expected_time:
                raise ValueError(
                    f"{txt_path}:{line_no}: row time {time_str} does not "
                    f"continue the previous rows (expected "
                    f"{timedelta(seconds=expected_time.total_seconds() % 86400)})"
                )
            if duration != EPOCH_DURATION:
                raise ValueError(
                    f"{txt_path}:{line_no}: duration {duration} != {EPOCH_DURATION}"
                )
            if stage_str not in STAGE_MAP:
                raise ValueError(
                    f"{txt_path}:{line_no}: unknown stage {stage_str!r}, "
                    f"expected one of {sorted(STAGE_MAP)}"
                )
            stages.append(STAGE_MAP[stage_str])
            expected_time = (time_sec + timedelta(seconds=duration)) % timedelta(days=1)
    if first_time is None:
        raise ValueError(f"{txt_path}: no label rows found")
    return first_time, np.asarray(stages, dtype=np.int64)


def apply_filtering(psg, sfreq):
    """
    Bandpass-filters all channels between FILTER_L_FREQ and FILTER_H_FREQ.

    Equivalent to psg_utils.preprocessing.filtering.apply_filtering with
    {'l_freq': FILTER_L_FREQ, 'h_freq': FILTER_H_FREQ, 'method': 'fir'}: mne
    FIR filter on float64, run at the original sample rate before resampling
    (the psg_utils _load pipeline filters before set_psg_sample_rate).
    """
    return filter_data(
        psg.T.astype(np.float64),
        sfreq,
        l_freq=FILTER_L_FREQ,
        h_freq=FILTER_H_FREQ,
        method="fir",
    ).T.astype(psg.dtype)


def clip_noisy_values(psg, sample_rate, period_length_sec, subject):
    """
    Clips all values larger/smaller than +/- MIN_MAX_TIMES_GLOBAL_IQR times the
    IQR of the whole channel, logging the affected epochs per channel.

    Equivalent to psg_utils.preprocessing.quality_control_funcs.clip_noisy_values
    with min_max_times_global_iqr=20 (preprocess.yaml default).
    """
    if np.any(~np.isfinite(psg)):
        raise ValueError(f"{subject}: PSG contains NaN or infinite values")
    n_periods = int(psg.shape[0] / (sample_rate * period_length_sec))
    for chan in range(psg.shape[-1]):
        chan_psg = psg[:, chan]
        iqr = np.subtract(*np.percentile(chan_psg, [75, 25]))
        threshold = iqr * MIN_MAX_TIMES_GLOBAL_IQR
        temp_psg = chan_psg.reshape(n_periods, -1)
        inds = np.unique(np.where(np.abs(temp_psg) > threshold)[0])
        psg[:, chan] = np.clip(chan_psg, -threshold, threshold)
        fraction = len(inds) / n_periods
        msg = (
            f"Quality control for '{subject}' affected {len(inds)}/{n_periods} "
            f"epochs in channel {chan}"
        )
        logger.warning(msg) if fraction >= 0.15 else logger.debug(msg)


def preprocess_psg(psg, sfreq, subject):
    """
    Trims to whole 30 s epochs, bandpass-filters at the original rate,
    resamples to 128 Hz, clips to +/- 20 x IQR and robust-scales each channel
    on the whole recording (training pipeline).

    Returns:
        psg:      float32 ndarray [n_epochs * SAMPLE_RATE * EPOCH_DURATION, C]
        n_epochs: number of complete 30 s epochs
    """
    data_per_epoch = sfreq * EPOCH_DURATION
    n_epochs = psg.shape[0] // data_per_epoch
    if n_epochs == 0:
        raise ValueError(
            f"{subject}: recording shorter than one {EPOCH_DURATION} s epoch"
        )
    psg = psg[: n_epochs * data_per_epoch]

    # bandpass filter at the original sample rate, before resampling
    # (psg_utils applies filter_settings before set_psg_sample_rate)
    psg = apply_filtering(psg, sfreq)

    # resample to SAMPLE_RATE (psg_utils set_psg_sample_rate, poly method)
    psg = resample_poly(psg, SAMPLE_RATE, sfreq, axis=0)

    # quality control, then per-channel RobustScaler fit on the whole recording
    clip_noisy_values(psg, SAMPLE_RATE, EPOCH_DURATION, subject)
    for chan in range(psg.shape[-1]):
        scaler = RobustScaler()  # with_centering=True, quantile_range (25, 75)
        psg[:, chan] = scaler.fit_transform(psg[:, chan].reshape(-1, 1)).ravel()

    return psg.astype(np.float32), n_epochs
