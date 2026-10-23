"""
Create the AnySleep training dataset 'bci2026' from the raw BCI recordings.

Replicates the preprocessing pipeline of dslaborg/Sleep-Datasets (preprocess.py
with the standard config/dataset settings) for the simple case of this dataset:

    load PSG (channels already referenced, no montage creation)
    -> strip_to_match (trim PSG/hypnogram to common whole-epoch lengths)
    -> bandpass 0.5-16 Hz FIR at the original rate       (psg_utils filtering)
    -> resample_poly(psg, 128, 250)                       (psg_utils psg_sampling)
    -> clip_noisy_values (clip to +/- 20x global IQR)      (quality_control_funcs)
    -> RobustScaler (per channel, fit on the whole record) (MultiChannelScaler)
    -> float32 -> single HDF5 file

Raw data (verified for all recordings):
    <subject>.edf  channels 'EEG', 'EOG-R', 'EOG-L', 'M1', 'ECG', 250 Hz
    <subject>.txt  tab-separated '时间 [hh:mm:ss] / 睡眠期 / 持续时间', 30 s rows,
                   stages W/N1/N2/N3/REM, first row aligned with the EDF start time
M1 is stored as a second EEG-group channel: the U-Sleep train sampler draws
one random EEG ('EEG' or 'M1') and one random EOG channel per window
(base/data/usleep_train_dataset.py). EOG-L and ECG are not used.

Outputs (next to this script):
    bci2026.h5          group 'bci2026' with one subgroup per recording, in the
                        structure expected by AnySleep (see base/data/base_dataset.py)
    bci2026_split.yaml  train/valid/test datasplit (prepare_splits.py semantics,
                        seed=42, test 2%, valid 18%)

Run with the anysleep conda env:
    python prepare_bci2026.py [--input-dir ...] [--overwrite] [--split-only]
"""
import argparse
import logging
import math
import os
import re
from datetime import datetime, timedelta
from glob import glob

import h5py
import numpy as np
from mne.filter import filter_data
import pyedflib
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)
logging.captureWarnings(True)

DATASET_NAME = "bci2026"
SAMPLE_RATE = 128  # output sample rate of the signals, in Hz
EPOCH_DURATION = 30  # sleep epoch duration, in seconds
EEG_CHANNELS = ["EEG", "M1"]
EOG_CHANNELS = ["EOG-R"]
# bandpass filter cutoffs (Hz), applied at the original sample rate before
# resampling, as the psg_utils _load pipeline does (filtering.apply_filtering)
FILTER_L_FREQ = 0.5
FILTER_H_FREQ = 16
# stage encoding of Sleep-Datasets (general.stages / general.rstages)
STAGE_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
MIN_MAX_TIMES_GLOBAL_IQR = 20  # quality control threshold, see preprocess.yaml
# split settings, see config/dataset/_base_dataset.yaml
TEST_FRACTION = 0.00
VALIDATION_FRACTION = 0.10
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = r"C:\Users\fpan\Desktop\2_有标签训练数据50例"


def parse_hypno_file(txt_path):
    """
    Parses a label file into per-epoch sleep stage integers.

    Validates that all rows are EPOCH_DURATION long, contiguous (row times
    equal the first time plus the cumulative durations, modulo 24 h) and use
    known stage names.

    Returns:
        first_time:   datetime of the first row
        stages:       int ndarray of shape [n_epochs]
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
            time_str, stage_str, dur_str = parts
            if first_time is None and not re.fullmatch(r"\d{2}:\d{2}:\d{2}", time_str.strip()):
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


def load_psg(edf_path):
    """
    Loads the EEG/EOG channels of an EDF file.

    Returns:
        psg:    float64 ndarray of shape [N, len(EEG_CHANNELS + EOG_CHANNELS)]
        sfreq:  sample rate of the file, in Hz
        start:  datetime of the recording start
    """
    channels = EEG_CHANNELS + EOG_CHANNELS
    f = pyedflib.EdfReader(edf_path)
    try:
        labels = [f.getLabel(i) for i in range(f.signals_in_file)]
        duplicates = sorted(chan for chan in set(labels) if labels.count(chan) > 1)
        if duplicates:
            raise ValueError(
                f"{edf_path}: duplicate channel labels {duplicates} ({labels})"
            )
        missing = [chan for chan in channels if chan not in labels]
        if missing:
            raise ValueError(
                f"{edf_path}: channels {missing} not found in file ({labels})"
            )
        # load the channels in the requested order, independent of the file order
        label_to_ind = {chan: i for i, chan in enumerate(labels)}
        sfreqs = {
            round(f.getSampleFrequency(label_to_ind[chan]), 10) for chan in channels
        }
        if len(sfreqs) != 1:
            raise ValueError(
                f"{edf_path}: selected channels have mixed/unknown sample rates: {sfreqs}"
            )
        sfreq = int(sfreqs.pop())
        psg = np.stack([f.readSignal(label_to_ind[chan]) / 1.0e6 for chan in channels], axis=1)
        start = f.getStartdatetime()
    finally:
        f.close()
    return np.ascontiguousarray(psg, dtype=np.float64), sfreq, start


def apply_filtering(psg, sfreq):
    """
    Bandpass-filters all channels between FILTER_L_FREQ and FILTER_H_FREQ.

    Equivalent to psg_utils.preprocessing.filtering.apply_filtering with
    {'l_freq': 0.5, 'h_freq': 16, 'method': 'fir'}: mne FIR filter on float64,
    run at the original sample rate before resampling (the psg_utils _load
    pipeline filters before set_psg_sample_rate).
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


def preprocess_recording(edf_path, txt_path, subject):
    """
    Runs the full preprocessing pipeline for one recording.

    Returns:
        psg:     float32 ndarray [n_epochs * SAMPLE_RATE * EPOCH_DURATION, C]
        stages:  int64 ndarray [n_epochs]
    """
    psg, sfreq, edf_start = load_psg(edf_path)
    hypno_start, stages = parse_hypno_file(txt_path)
    if edf_start.strftime("%H:%M:%S") != hypno_start.strftime("%H:%M:%S"):
        raise ValueError(
            f"{subject}: hypnogram starts at {hypno_start:%H:%M:%S} but the EDF "
            f"starts at {edf_start:%H:%M:%S}; the labels would be misaligned"
        )

    # strip_to_match: trim the PSG to a length divisible by EPOCH_DURATION * sfreq
    # and align the hypnogram length with it (pad with UNKNOWN=9 / truncate)
    data_per_epoch = sfreq * EPOCH_DURATION
    n_epochs = psg.shape[0] // data_per_epoch
    psg = psg[: n_epochs * data_per_epoch]
    if len(stages) > n_epochs:
        logger.info(f"{subject}: stripping hypnogram by {len(stages) - n_epochs} epochs")
        stages = stages[:n_epochs]
    elif len(stages) < n_epochs:
        logger.info(f"{subject}: padding hypnogram with {n_epochs - len(stages)} UNKNOWN epochs")
        stages = np.pad(stages, (0, n_epochs - len(stages)), constant_values=9)

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

    return psg.astype(np.float32), stages


def write_subject(h5_dataset_group, subject, psg, stages):
    """Writes one preprocessed recording in the structure expected by AnySleep."""
    study_group = h5_dataset_group.create_group(subject)
    psg_group = study_group.create_group("PSG")
    n_periods = len(stages)
    data_per_period = SAMPLE_RATE * EPOCH_DURATION
    X = psg.reshape(n_periods, data_per_period, -1)
    for chan_ind, channel_name in enumerate(EEG_CHANNELS + EOG_CHANNELS):
        psg_group.create_dataset(channel_name, data=X[..., chan_ind].ravel())
    study_group.create_dataset("hypnogram", data=stages)
    cls_to_indx_group = study_group.create_group("class_to_index")
    dtype = np.dtype("uint16") if n_periods <= 65535 else np.dtype("uint32")
    for class_ in np.unique(stages):
        inds = np.where(stages == class_)[0].astype(dtype)
        cls_to_indx_group.create_dataset(str(class_), data=inds)
    study_group.attrs["sample_rate"] = SAMPLE_RATE


def dict_to_yaml(value_to_write, depth=0):
    """Writes the datasplit dict in the format of prepare_splits.py."""
    yaml_string = ""
    whitespaces = "  " * depth
    if isinstance(value_to_write, dict):
        for key, value in value_to_write.items():
            yaml_string += f"\n{whitespaces}{key}: {dict_to_yaml(value, depth+1)}"
    elif isinstance(value_to_write, list):
        for value in value_to_write:
            yaml_string += f"\n{whitespaces}- {dict_to_yaml(value, depth+1)}"
    elif isinstance(value_to_write, str):
        yaml_string += f'"{value_to_write}"'
    else:
        raise NotImplementedError(f"Type {type(value_to_write)} not implemented")
    return yaml_string


def make_split(subjects, save_file):
    """
    Creates the train/valid/test datasplit with the rules of prepare_splits.py
    (cv_n_splits=1): ceil-fractional test/valid sizes, RandomState(SEED) shuffle,
    test drawn first, then valid from the remainder.
    """
    n_total = len(subjects)
    n_test = int(math.ceil(n_total * TEST_FRACTION))
    n_val = int(math.ceil(n_total * VALIDATION_FRACTION))
    if n_val + n_test > n_total:
        raise ValueError(
            "Too large test/validation_fraction - fractions must be <= 1.0 in total"
        )
    randomstate = np.random.RandomState(SEED)
    subject_dirs = np.array(sorted(subjects))
    randomstate.shuffle(subject_dirs)
    test_data = subject_dirs[:n_test]
    train_val_data = subject_dirs[n_test:]
    randomstate.shuffle(train_val_data)
    valid_data = train_val_data[:n_val]
    train_data = train_val_data[n_val:]

    datasplit_dict = {}
    datasplit_dict["train"] = {DATASET_NAME: [str(s) for s in train_data]}
    datasplit_dict["valid"] = {DATASET_NAME: [str(s) for s in valid_data]}
    datasplit_dict["test"] = {DATASET_NAME: [str(s) for s in test_data]}
    with open(save_file, "w") as f:
        f.write(dict_to_yaml(datasplit_dict))
    logger.info(
        f"Saved datasplit ({n_total} subjects: {len(train_data)} train / "
        f"{len(valid_data)} valid / {len(test_data)} test) to {save_file}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", default=SCRIPT_DIR)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing output files (they are only "
                             "rewritten when the run reaches them, so a failed "
                             "run does not delete previous outputs)")
    parser.add_argument("--split-only", action="store_true",
                        help="only write the datasplit yaml from the EDF file "
                             "names, without regenerating the h5 file")
    args = parser.parse_args()

    out_h5 = os.path.join(args.out_dir, f"{DATASET_NAME}.h5")
    out_split = os.path.join(args.out_dir, f"{DATASET_NAME}_split.yaml")
    outputs = (out_split,) if args.split_only else (out_h5, out_split)
    for path in outputs:
        if os.path.exists(path) and not args.overwrite:
            raise OSError(
                f"File already exists at '{path}' (delete it or pass "
                f"--overwrite to re-run)"
            )

    edf_files = sorted(glob(os.path.join(args.input_dir, "*.edf")))
    logger.info(f"Found {len(edf_files)} EDF files matching glob statement")
    if len(edf_files) == 0:
        raise ValueError(f"No .edf files found in '{args.input_dir}'")

    subjects = [os.path.splitext(os.path.basename(f))[0] for f in edf_files]
    if args.split_only:
        make_split(subjects, out_split)
        return

    txt_files = sorted(glob(os.path.join(args.input_dir, "*.txt")))
    if len(edf_files) != len(txt_files):
        raise ValueError(
            f"Expected equal numbers of .edf and .txt files in "
            f"'{args.input_dir}', got {len(edf_files)} and {len(txt_files)}"
        )

    stage_names = {v: k for k, v in STAGE_MAP.items()}
    total_stages = np.zeros(5, dtype=np.int64)
    with h5py.File(out_h5, "w") as h5_file:
        h5_dataset_group = h5_file.create_group(DATASET_NAME)
        h5_dataset_group.attrs["eeg_channels"] = EEG_CHANNELS
        h5_dataset_group.attrs["eog_channels"] = EOG_CHANNELS
        h5_dataset_group.attrs["emg_channels"] = []

        for i, edf_path in enumerate(edf_files):
            subject = os.path.splitext(os.path.basename(edf_path))[0]
            logger.info("------------------")
            logger.info(f"[*] {i+1}/{len(edf_files)} Processing {subject}")
            psg, stages = preprocess_recording(
                edf_path, os.path.splitext(edf_path)[0] + ".txt", subject
            )
            write_subject(h5_dataset_group, subject, psg, stages)
            logger.info(
                f"[*] Extracted {psg.shape[1]} channels, {len(stages)} epochs"
            )
            for cls in range(5):
                total_stages[cls] += np.sum(stages == cls)

    make_split(subjects, out_split)

    logger.info("")
    logger.info(f"[*] Dataset written to {out_h5}")
    stage_summary = ", ".join(
        f"{stage_names[cls]}: {total_stages[cls]}" for cls in range(5)
    )
    logger.info(f"[*] Total epochs per stage: {stage_summary}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(module)s:%(lineno)d] - %(message)s",
    )
    main()
