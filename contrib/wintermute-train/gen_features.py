#!/usr/bin/env python3
"""Generate spectrogram feature sets for micro-wake-word training.

Faithful to OHF-Voice/micro-wake-word basic_training_notebook cells 4–8:
 - download MIT RIRs (+ optional audioset/fma background) for augmentation
 - build positive spectrogram features from the generated "wintermute" wavs
 - download pre-generated NEGATIVE spectrogram features from HuggingFace
   (kahrendt/microwakeword: speech / no_speech / dinner_party)

--smoke 1 keeps everything tiny so the whole pipeline can be validated fast.
"""
import argparse, os, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path


def log(*a): print("[gen_features]", *a, flush=True)


def download(url, dest):
    if os.path.exists(dest):
        return
    log("download", url, "->", dest)
    # Prefer curl: it follows HuggingFace's xet-bridge 302 redirects and
    # supports resume (-C -), which urllib.urlretrieve does not do reliably
    # for the multi-GB negative-dataset zips (a partial urllib download is
    # what produced the truncated no_speech.zip).
    if shutil.which("curl"):
        rc = subprocess.call(["curl", "-fL", "-C", "-", "-o", str(dest), url])
        if rc != 0:
            log("curl rc", rc, "— falling back to urllib")
            urllib.request.urlretrieve(url, dest)
    else:
        urllib.request.urlretrieve(url, dest)


def fetch_rirs(root, smoke):
    import datasets, scipy.io.wavfile, numpy as np
    out = root / "mit_rirs"
    if out.exists():
        return out
    out.mkdir(parents=True)
    ds = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train", streaming=True)
    n = 0
    cap = 15 if smoke else 10_000
    for row in ds:
        a = row["audio"]
        scipy.io.wavfile.write(out / f"{n}.wav", 16000,
                               (a["array"] * 32767).astype(np.int16))
        n += 1
        if n >= cap:
            break
    log("RIRs:", n)
    return out


def fetch_negative_features(root, smoke):
    """Pre-generated negative spectrogram feature sets (RaggedMmap zips)."""
    out = root / "negative_datasets"
    out.mkdir(parents=True, exist_ok=True)
    link_root = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
    # smoke: just 'no_speech' (smallest) to prove the wiring; full: all four.
    names = ["no_speech.zip"] if smoke else \
            ["speech.zip", "no_speech.zip", "dinner_party.zip", "dinner_party_eval.zip"]
    for fname in names:
        zp = out / fname
        marker = out / fname.replace(".zip", "")
        if marker.exists():
            continue
        # (Re)download if the local copy is missing or not a valid zip.
        need_dl = not zp.exists()
        if zp.exists():
            if not zipfile.is_zipfile(zp):
                log("WARN: existing", fname, "is not a valid zip (truncated?) — re-downloading")
                zp.unlink()
                need_dl = True
        if need_dl:
            download(link_root + fname, zp)
        if not zipfile.is_zipfile(zp):
            log("ERROR:", fname, "is still not a valid zip after download")
            sys.exit(1)
        log("unzip", fname)
        with zipfile.ZipFile(zp) as z:
            z.extractall(out)
    return out


def build_positive_features(gen_dir, root, smoke):
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    rir_dir = fetch_rirs(root, smoke)
    out_dir = root / "generated_augmented_features"
    out_dir.mkdir(exist_ok=True)

    clips = Clips(input_directory=str(gen_dir), file_pattern="*.wav",
                  max_clip_duration_s=None, remove_silence=False,
                  random_split_seed=10, split_count=0.1)
    aug = Augmentation(
        augmentation_duration_s=3.2,
        # Anti-overfit (real-voice retrain): stronger/broader augmentation so the
        # model learns the word's invariants, not the few captured utterances.
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.3, "TanhDistortion": 0.2,
            "PitchShift": 0.4, "BandStopFilter": 0.2, "AddColorNoise": 0.3,
            "AddBackgroundNoise": 0.75, "Gain": 1.0, "RIR": 0.5,
        },
        impulse_paths=[str(rir_dir)],
        background_paths=[str(rir_dir)],  # smoke: reuse RIRs as bg; full run adds audioset/fma
    )
    for split, split_name, rep, slide in [
        ("training", "train", 8, 10),   # was 2: 4x more augmented variants per clip
        ("validation", "validation", 1, 10),
        ("testing", "test", 1, 1),
    ]:
        d = out_dir / split
        d.mkdir(exist_ok=True)
        sg = SpectrogramGeneration(clips=clips, augmenter=aug,
                                   slide_frames=slide, step_ms=10)
        RaggedMmap.from_generator(
            out_dir=str(d / "wakeword_mmap"),
            sample_generator=sg.spectrogram_generator(split=split_name, repeat=rep),
            batch_size=100, verbose=True)
        log("built", split)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True)
    ap.add_argument("--positives", required=True, help="dir of generated positive wavs")
    ap.add_argument("--out", required=True, help="feature output root")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    smoke = bool(args.smoke)

    n_pos = len(list(Path(args.positives).glob("*.wav")))
    if n_pos == 0:
        log("ERROR: no positive wavs in", args.positives)
        sys.exit(1)
    log(f"positives: {n_pos} wavs; smoke={smoke}")

    build_positive_features(Path(args.positives), root, smoke)
    fetch_negative_features(root, smoke)
    log("feature generation complete ->", root)


if __name__ == "__main__":
    main()
