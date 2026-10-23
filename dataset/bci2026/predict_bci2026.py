"""
Batch sleep-stage inference for the BCI 2026 competition.

Reads every EDF file from a (read-only) input directory, predicts one sleep
stage per 30 s epoch with the U-Sleep model (2-channel EEG + EOG input), and
writes one same-named UTF-8 JSON file per EDF into the output directory:

    {"Sleep": [{"time": "2026-01-01 23:59:30", "period": 0}, ...]}

    period: W=0, N1=1, N2=2, N3=3, REM=4
    time:   epoch start = EDF start time + 30 s * epoch index
            (the date rolls over across midnight)

Each EDF is processed completely independently: channels are selected per
file and the whole recording is preprocessed on its own. No model state,
no trimming, and no statistics are carried across records. The raw argmax
hypnograms are then repaired by rule-based post-processing (see
postprocess_periods), enabled by default; --no-postprocess skips it.

Preprocessing (identical to prepare_bci2026.py and the training pipeline,
same as examples/predict_edf_file_logits_plain.py):
    EEG + EOG-R read via pyedflib (physical calibration from the EDF
    header; the signals are already 0.1-35 Hz bandpassed and 50 Hz notched;
    M1 and ECG are not used, and EOG-L is available in the EDFs but not
    loaded: U-Sleep takes exactly 2 channels, EEG first and EOG second,
    the same pair as the exp001a U-Sleep training data in
    prepare_bci2026.py)
    -> trim to whole 30 s epochs from the EDF start
    -> resample_poly to 128 Hz (model input rate)
    -> clip to +/- 20 x global IQR per channel
    -> per-channel RobustScaler (median/IQR, fit on the whole recording)
    -> float32, whole-recording forward pass (uses the full context)

Usage (anysleep conda env; torch is required only for inference):
    python predict_bci2026.py --input-dir /input --output-dir /output
    python predict_bci2026.py --input-dir <edf dir> --output-dir <json dir> \
        --truth-dir <label txt dir>          # infer + print competition metrics
    python predict_bci2026.py --score-only --output-dir <json dir> \
        --truth-dir <label txt dir>          # score existing JSONs, no model

Label TXT layout (as in the training data): tab-separated
'时间 [hh:mm:ss] / 睡眠期 / 持续时间' with 30 s rows aligned to the EDF start.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyedflib
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler

try:  # torch is only needed for inference, not for --score-only
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../AnySleep

# default checkpoint: the U-Sleep model published with the repo (the same
# weights used by examples/anphy/predict_anphy_usleep.py). Pass --model to
# use another U-Sleep checkpoint, e.g. a bci2026-trained one saved into
# bci2026_models/ next to the repo as exp001a-usleep-<timestamp>-final.pth
DEFAULT_MODEL = REPO_ROOT.parent / "bci2026_models" / "exp001a-usleep-2026-10-23_10-52-18-final.pth"

# U-Sleep input: exactly 2 channels, (EEG, EOG) in this order
EEG_CHANNELS = ["EEG"]
EOG_CHANNELS = ["EOG-R"]
SAMPLE_RATE = 128  # model input sample rate, Hz
EPOCH_DURATION = 30  # sleep epoch duration, s
MIN_MAX_TIMES_GLOBAL_IQR = 20  # quality control threshold
STAGE_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def load_psg(edf_path):
    """
    Loads the EEG/EOG channels of an EDF file (physical values, calibrated
    from the EDF header digital/physical min-max).

    Returns:
        psg:    float64 ndarray of shape [N, len(EEG_CHANNELS + EOG_CHANNELS)]
        sfreq:  sample rate of the file, in Hz
        start:  naive datetime of the recording start, from the EDF header
    """
    channels = EEG_CHANNELS + EOG_CHANNELS
    f = pyedflib.EdfReader(str(edf_path))
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
                f"{edf_path}: selected channels have mixed/unknown sample rates: "
                f"{sfreqs}"
            )
        sfreq = int(sfreqs.pop())
        psg = np.stack([f.readSignal(label_to_ind[chan]) / 1.0e6 for chan in channels], axis=1)
        start = f.getStartdatetime()
    finally:
        f.close()
    if start is None or start.year < 1980:
        raise ValueError(f"{edf_path}: EDF header has no valid start date/time")
    return np.ascontiguousarray(psg, dtype=np.float64), sfreq, start


def preprocess_recording(psg, sfreq):
    """
    Trims to whole 30 s epochs, resamples to 128 Hz, clips to +/- 20 x IQR and
    robust-scales each channel on the whole recording (training pipeline).

    Returns:
        psg:      float32 ndarray [n_epochs * SAMPLE_RATE * EPOCH_DURATION, C]
        n_epochs: number of complete 30 s epochs
    """
    data_per_epoch = sfreq * EPOCH_DURATION
    n_epochs = psg.shape[0] // data_per_epoch
    if n_epochs == 0:
        raise ValueError(f"recording shorter than one {EPOCH_DURATION} s epoch")
    psg = psg[: n_epochs * data_per_epoch]

    psg = resample_poly(psg, SAMPLE_RATE, sfreq, axis=0)

    for chan in range(psg.shape[-1]):
        chan_psg = psg[:, chan]
        iqr = np.subtract(*np.percentile(chan_psg, [75, 25]))
        threshold = iqr * MIN_MAX_TIMES_GLOBAL_IQR
        psg[:, chan] = np.clip(chan_psg, -threshold, threshold)

    for chan in range(psg.shape[-1]):
        scaler = RobustScaler()  # with_centering=True, quantile_range (25, 75)
        psg[:, chan] = scaler.fit_transform(psg[:, chan].reshape(-1, 1)).ravel()

    return psg.astype(np.float32), n_epochs


def predict_edf(model, edf_path, device):
    """Runs the model on one EDF file, using its whole recording as context.

    Returns:
        start:    recording start datetime (EDF header)
        periods:  int ndarray [n_epochs], W=0 .. REM=4 (model class order,
                  same encoding as the training labels)
    """
    psg, sfreq, start = load_psg(edf_path)
    psg, n_epochs = preprocess_recording(psg, sfreq)
    x = torch.from_numpy(psg).float().unsqueeze(0).to(device)  # (1, time, ch)
    with torch.no_grad():
        logits = model(x)  # (1, epochs, 5)
    periods = logits[0].argmax(dim=1).cpu().numpy().astype(np.int64)
    if len(periods) != n_epochs:
        raise ValueError(
            f"{edf_path}: model returned {len(periods)} predictions for "
            f"{n_epochs} epochs"
        )
    return start, periods


def write_prediction_json(out_path, start, periods):
    """Writes one competition JSON: epoch start times and integer stages."""
    sleep = [
        {
            "time": (start + timedelta(seconds=i * EPOCH_DURATION)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "period": int(period),
        }
        for i, period in enumerate(periods)
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"Sleep": sleep}, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_bounds(arr):
    """Start/end indices of the maximal same-stage runs of arr."""
    change = np.flatnonzero(np.diff(arr) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(arr)]])
    return starts, ends


def rule_n3_rem_break(pred, max_len=2):
    """Short N3 runs adjacent to REM become N2."""
    pred = pred.copy()
    starts, ends = run_bounds(pred)
    for j in range(len(starts)):
        s, e = starts[j], ends[j]
        if pred[s] == 3 and e - s <= max_len:
            before = pred[s - 1] if s > 0 else None
            after = pred[e] if e < len(pred) else None
            if 4 in (before, after):
                pred[s:e] = 2
    return pred


def rule_w_adj_rem(pred, max_len=2):
    """W runs of <= max_len epochs with a REM neighbor on one side and a REM
    or N2 neighbor on the other become REM (generalizes rule_w_in_rem: short
    W epochs hugging a REM run are usually truly REM also when the other side
    is N2, i.e. at REM-period boundaries; with W or N1 on the other side the
    truth split is ~50/50 and no repair pays off)."""
    pred = pred.copy()
    starts, ends = run_bounds(pred)
    for j in range(1, len(starts) - 1):
        s, e = starts[j], ends[j]
        if pred[s] == 0 and e - s <= max_len:
            neighbors = {int(pred[starts[j - 1]]), int(pred[ends[j]])}
            if neighbors in ({4}, {2, 4}):
                pred[s:e] = 4
    return pred


def rule_early_rem(pred, cutoff_min=30):
    """REM epochs before cutoff_min from the recording start become W. Truth
    REM latency is >= 31.5 min on all 50 local records, and every predicted
    REM epoch in the first 30 min is wrong (78% W, 18% N1)."""
    pred = pred.copy()
    cut = int(cutoff_min * 60 // EPOCH_DURATION)
    starts, ends = run_bounds(pred)
    for j in range(len(starts)):
        s, e = starts[j], ends[j]
        if pred[s] == 4 and s < cut:
            pred[s: min(e, cut)] = 0
    return pred


def rule_n3_in_n2(pred, max_len=1):
    """N3 runs of <= max_len epochs between N2 runs become N2 (isolated
    single-epoch N3 is more often truly N2, but the margin is thin)."""
    pred = pred.copy()
    starts, ends = run_bounds(pred)
    for j in range(1, len(starts) - 1):
        s, e = starts[j], ends[j]
        if (
            pred[s] == 3
            and e - s <= max_len
            and pred[starts[j - 1]] == 2
            and pred[ends[j]] == 2
        ):
            pred[s:e] = 2
    return pred


def rule_onset_w(pred, cutoff_min=5):
    """Non-W epochs before cutoff_min from the recording start become W.
    Truth sleep-onset latency is >= 3 min on all 50 local records (p10 8 min),
    while the model calls sleep at minute 0; predicted sleep in the first
    5 min is 87% truly W."""
    pred = pred.copy()
    cut = int(cutoff_min * 60 // EPOCH_DURATION)
    head = pred[:cut]
    head[head != 0] = 0
    pred[:cut] = head
    return pred


def rule_end_w(pred, max_len=2):
    """A final sleep run of <= max_len epochs preceded by W becomes W
    (recordings end with lights-on wake; short trailing N1/N2/REM fragments
    flanked by W are noise)."""
    pred = pred.copy()
    starts, ends = run_bounds(pred)
    if len(starts) < 2:
        return pred
    s, e = starts[-1], ends[-1]
    if pred[s] != 0 and e - s <= max_len and pred[starts[-2]] == 0:
        pred[s:e] = 0
    return pred


def postprocess_periods(periods):
    """
    Applies the rule-based repairs mined from the truth labels
    (analyze_truth_rules.py, analyze_transitions.py, analyze_durations.py)
    to one predicted hypnogram:

    - onset_w: truth sleep-onset latency is >= 3 min (p10 8 min) on all 50
      local records while the model calls sleep at minute 0; non-W epochs
      in the first 6 min become W (~87% truly W);
    - w_adj_rem: the model over-produces W<->REM transitions (4x the truth
      rate), splitting REM periods with W; W runs of <= 2 epochs hugging a
      REM run (other side REM or N2) are mostly truly REM, so they are
      relabeled to REM; with W or N1 on the other side the truth split is
      ~50/50 and no repair pays off;
    - n3_rem_break: N3<->REM never occurs in truth (deep sleep is entered
      and left through N2), so short N3 runs adjacent to REM become N2;
    - n3_in_n2: isolated single-epoch N3 runs between N2 runs become N2;
    - early_rem: truth REM latency is >= 31.5 min on all 50 local records
      and every predicted REM epoch in the first 30 min is wrong, so those
      become W;
    - end_w: a final sleep run of <= 2 epochs preceded by W becomes W
      (recordings end with lights-on wake).

    Together ~+1.0 score on the 50 local records, ~+1.3 on the 10 that were
    not in the model's training split. Truth has no hard minimum stage
    duration (many 1-epoch W/N1/N3 runs) and the model produces no runs
    longer than truth's observed maximums, so neither blanket minimum nor
    maximum duration rules pay off beyond these night-boundary cases.
    """
    fixed = rule_onset_w(periods, cutoff_min=6)
    fixed = rule_w_adj_rem(fixed)
    fixed = rule_n3_rem_break(fixed)
    fixed = rule_n3_in_n2(fixed)
    fixed = rule_early_rem(fixed)
    return rule_end_w(fixed)


def load_hypnogram(txt_path):
    """
    Parses a label TXT into per-epoch sleep stage integers.

    Expects tab-separated 'hh:mm:ss / stage / duration' rows of 30 s,
    contiguous from the first row, aligned with the EDF start time.

    Returns:
        first_time: datetime of the first row
        stages:     int64 ndarray of shape [n_epochs]
    """
    stages = []
    first_time = None
    expected_sec = None  # clock seconds of the next expected row
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
                continue  # header row
            if stage_str not in STAGE_MAP:
                raise ValueError(
                    f"{txt_path}:{line_no}: unknown stage {stage_str!r}, "
                    f"expected one of {sorted(STAGE_MAP)}"
                )
            if int(dur_str) != EPOCH_DURATION:
                raise ValueError(
                    f"{txt_path}:{line_no}: duration {dur_str} != {EPOCH_DURATION}"
                )
            sec = (
                int(time_str[:2]) * 3600 + int(time_str[3:5]) * 60 + int(time_str[6:8])
            )
            if first_time is None:
                first_time = datetime.strptime(time_str, "%H:%M:%S")
                expected_sec = sec
            elif sec != expected_sec % 86400:
                raise ValueError(
                    f"{txt_path}:{line_no}: row time {time_str} does not continue "
                    f"the previous rows"
                )
            stages.append(STAGE_MAP[stage_str])
            expected_sec += EPOCH_DURATION
    if first_time is None:
        raise ValueError(f"{txt_path}: no label rows found")
    return first_time, np.asarray(stages, dtype=np.int64)


def print_metrics(y_true, y_pred):
    """
    Prints the competition metrics on pooled epochs:
    per-class and macro F1, unweighted Cohen's kappa (not truncated at 0),
    accuracy, and the final score (F1 x 0.5 + Kappa x 0.3 + Acc x 0.2) x 100,
    rounded to 2 decimals.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)

    acc = float((y_true == y_pred).mean())

    f1_per_class = []
    for cls, name in enumerate(STAGE_NAMES):
        n_true = int((y_true == cls).sum())
        n_pred = int((y_pred == cls).sum())
        n_correct = int(((y_true == cls) & (y_pred == cls)).sum())
        denom = n_true + n_pred
        f1 = 2 * n_correct / denom if denom > 0 else 0.0
        f1_per_class.append(f1)
        print(f"  F1({name:>3}) = {f1:.4f}   true {n_true:5d}   pred {n_pred:5d}")
    macro_f1 = float(np.mean(f1_per_class))

    pe = float(
        sum((y_true == c).mean() * (y_pred == c).mean() for c in range(len(STAGE_NAMES)))
    )
    if pe == 1.0:  # both distributions concentrated on one class
        kappa = 1.0 if acc == 1.0 else 0.0
    else:
        kappa = (acc - pe) / (1.0 - pe)

    score = round((macro_f1 * 0.5 + kappa * 0.3 + acc * 0.2) * 100, 2)

    print(f"  Epochs  {n}")
    print(f"  Acc     {acc:.4f}")
    print(f"  Kappa   {kappa:.4f}")
    print(f"  MacroF1 {macro_f1:.4f}")
    print(f"  Score   {score:.2f}")
    return {"acc": acc, "kappa": kappa, "macro_f1": macro_f1, "score": score}


def load_predictions(json_path):
    """Reads a prediction JSON into an int64 array of periods."""
    with open(json_path, encoding="utf-8") as f:
        sleep = json.load(f)["Sleep"]
    periods = np.asarray([entry["period"] for entry in sleep], dtype=np.int64)
    if np.any((periods < 0) | (periods > 4)):
        raise ValueError(f"{json_path}: invalid period values")
    return sleep[0]["time"], periods


def run_inference(args):
    """Batch inference: one independent JSON per EDF in the input dir."""
    if torch is None:
        sys.exit("error: torch is required for inference (missing in this env)")

    sys.path.insert(0, str(REPO_ROOT / "examples"))
    try:
        from usleep_no_hydra import USleep
    except ImportError:
        sys.exit(
            f"error: cannot import usleep_no_hydra; expected at "
            f"{REPO_ROOT / 'examples' / 'usleep_no_hydra.py'}"
        )

    device = args.device
    print(f"Model:   {args.model}")
    print(f"Device:  {device}")
    # U-Sleep consumes (batch, time, 2) with the 2 channels EEG + EOG
    # (EEG_CHANNELS + EOG_CHANNELS above) and outputs 5 logits per epoch in
    # the repo's stage encoding (config/base_config.yaml data.stages:
    # 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM), which is identical to STAGE_MAP, so
    # the argmax labels need no remapping.
    model = USleep(path=args.model, sleep_stage_frequency=1)
    model.eval().to(device)

    input_dir = Path(args.input_dir)
    edf_files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".edf"
    )
    if not edf_files:
        sys.exit(f"error: no EDF files found in {input_dir}")
    if args.limit:
        edf_files = edf_files[: args.limit]
    print(f"Input:   {input_dir} ({len(edf_files)} EDF files)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output:  {output_dir}\n")

    t_total = time.perf_counter()
    for i, edf_path in enumerate(edf_files, start=1):
        print(f"[{i}/{len(edf_files)}] {edf_path.name}")
        t_file = time.perf_counter()
        start, periods = predict_edf(model, edf_path, device)
        n_post = 0
        if not args.no_postprocess:
            fixed = postprocess_periods(periods)
            n_post = int((fixed != periods).sum())
            periods = fixed
        out_path = output_dir / (edf_path.stem + ".json")
        write_prediction_json(out_path, start, periods)
        elapsed = time.perf_counter() - t_file
        print(
            f"  {start:%Y-%m-%d %H:%M:%S} + {len(periods)} epochs -> "
            f"{out_path.name} ({elapsed:.1f} s, {n_post} epochs post-processed)"
        )

    total = time.perf_counter() - t_total
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    print(
        f"\nDone: {len(edf_files)} files in {int(h)}:{int(m):02d}:{s:04.1f} "
        f"(avg {total / len(edf_files):.1f} s/file)"
    )


def run_scoring(args):
    """Pools all epochs of all records and prints the competition metrics."""
    output_dir = Path(args.output_dir)
    json_files = sorted(output_dir.glob("*.json"))
    if not json_files:
        sys.exit(f"error: no JSON files found in {output_dir}")

    y_true_all, y_pred_all = [], []
    for json_path in json_files:
        txt_path = Path(args.truth_dir) / (json_path.stem + ".txt")
        if not txt_path.exists():
            sys.exit(f"error: no label file {txt_path}")
        first_time, stages = load_hypnogram(txt_path)
        first_time_str, periods = load_predictions(json_path)

        if first_time_str[11:] != first_time.strftime("%H:%M:%S"):
            print(
                f"warning: {json_path.stem}: predictions start at "
                f"{first_time_str[11:]} but labels start at "
                f"{first_time:%H:%M:%S}"
            )
        if len(stages) != len(periods):
            print(
                f"warning: {json_path.stem}: {len(periods)} predictions vs "
                f"{len(stages)} labels, scoring the first {min(len(stages), len(periods))}"
            )

        n = min(len(stages), len(periods))
        y_true_all.append(stages[:n])
        y_pred_all.append(periods[:n])

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    print(f"\nScoring {len(json_files)} recordings, {len(y_true)} pooled epochs\n")
    print_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(
        description="Batch sleep staging inference for BCI 2026 (EDF dir -> JSON dir)"
    )
    parser.add_argument("--input-dir", default="/input", help="read-only EDF directory")
    parser.add_argument("--output-dir", default="/output", help="JSON output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="U-Sleep checkpoint")
    parser.add_argument("--device", default="cpu", help="cuda or cpu (default: cpu)")
    parser.add_argument(
        "--truth-dir",
        default=None,
        help="label TXT directory; if set, prints the competition metrics",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="skip inference, only score existing output JSONs (needs --truth-dir)",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="write raw model predictions, skip the default rule-based repairs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="process only the first N EDF files (default: 0 = all of them)",
    )
    args = parser.parse_args()

    if args.score_only:
        if not args.truth_dir:
            sys.exit("error: --score-only needs --truth-dir")
        run_scoring(args)
        return

    run_inference(args)
    if args.truth_dir:
        run_scoring(args)


if __name__ == "__main__":
    main()
