"""
Vocal Biomarker API v5 — Calibrated from Real Log Data
========================================================

Fixes from v4 log analysis (growl recording):
----------------------------------------------
LOG FINDING 1: f0_median=0.0, n_voiced_frames=0
  PYIN found ZERO voiced frames in the growl. This means creaky_ratio
  falls to the 0.8 fallback instead of measuring actual creaky frames.
  But for normal speech, PYIN may also struggle with iPhone audio quality,
  giving artificially high creaky ratios.
  FIX: Lower PYIN fmin to 40 Hz (growls can be very low). Also, when
  n_voiced_frames=0, treat it as MAXIMUM strain (1.0, not 0.8) because
  a recording where no pitch can be detected is definitionally abnormal.
  For normal speech, add a "healthy pitch detected" bonus that reduces
  strain when PYIN successfully tracks a normal-range F0.

LOG FINDING 2: spectral_tilt_db=-2.45
  NEGATIVE tilt means more high-freq energy than low. This is caused by
  our pre-emphasis filter (+6dB/octave boost) overwhelming the natural
  low-frequency dominance of the growl. The growl's energy was below
  100 Hz which gets attenuated by the iPhone mic, then what little
  remains gets further shifted by pre-emphasis.
  FIX: Compute spectral tilt on the ORIGINAL signal (before pre-emphasis).
  Pre-emphasis is still useful for onset detection but must not affect
  tilt measurement.

LOG FINDING 3: n_onsets=13 in 5.55s of growling with barely any speech
  The growl's pulsing creates spectral flux peaks that pass the energy
  gate. 13 onsets → syllable_rate=2.34 → cog_speed=58.5.
  FIX: Add a VOICED-FRAME gate on top of energy gate. Onsets only count
  if they occur near frames where PYIN detected actual pitched speech.
  A growl with 0 voiced frames → 0 valid onsets.

NEW: PHRASE-RELATIVE COG SPEED
  The endpoint now accepts an optional `phrase` form field — the text
  shown on screen for the user to read. We estimate the expected syllable
  count from the phrase and compute completion_ratio = detected / expected.
  If no phrase is provided, we fall back to duration-based estimation.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vocal_biomarker")

app = FastAPI(title="Vocal Biomarker API", version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BiomarkerResult(BaseModel):
    vrs: float = Field(..., ge=0, le=100)
    tension: float = Field(..., ge=0, le=100)
    vitality: float = Field(..., ge=0, le=100)
    cog_speed: float = Field(..., ge=0, le=100)
    debug: Optional[dict] = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR = 22050
MIN_DURATION_SEC = 0.6
PRE_EMPHASIS_COEFF = 0.97

F0_CREAKY = {"male": 90, "female": 135, "other": 110}
F0_NORMAL_RANGE = {
    "male":   (85, 180),
    "female": (150, 300),
    "other":  (100, 240),
}

# Vowel-heavy languages average ~1.3-1.5 syllables per word.
# English averages ~1.4 syllables per word.
SYLLABLES_PER_WORD_EST = 1.4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float, mid: float, slope: float) -> float:
    z = np.clip(slope * (x - mid), -500, 500)
    return float(1.0 / (1.0 + np.exp(-z)))


def _estimate_syllable_count(phrase: str) -> int:
    """
    Rough syllable count from a text phrase.
    Uses a vowel-cluster heuristic: count groups of consecutive vowels.
    Falls back to word_count * 1.4 if the heuristic gives 0.
    """
    if not phrase or not phrase.strip():
        return 0
    # Remove non-alpha characters
    clean = re.sub(r"[^a-zA-ZàáâãäåèéêëìíîïòóôõöùúûüñçÀ-ÿ]", " ", phrase)
    words = clean.split()
    if not words:
        return 0

    total = 0
    for word in words:
        word = word.lower()
        # Count vowel clusters
        clusters = re.findall(r"[aeiouyàáâãäåèéêëìíîïòóôõöùúûü]+", word)
        count = len(clusters)
        # Every word has at least 1 syllable
        if count == 0:
            count = 1
        # Subtract silent-e at end (English heuristic)
        if word.endswith("e") and count > 1:
            count -= 1
        total += count

    return max(total, 1)


def _tension_curve(raw_strain: float) -> float:
    """
    0.00-0.10 → tension  2-10  (normal healthy voice)
    0.15-0.25 → tension 12-28  (slight strain)
    0.35-0.50 → tension 42-62  (moderate)
    0.60-0.80 → tension 75-93  (severe growl/fry)
    0.85+     → tension 95-99  (extreme)
    """
    s = _sigmoid(raw_strain, mid=0.40, slope=8.5)
    return float(np.clip(s * 100, 0, 100))


def _vitality_from_tension(tension: float) -> float:
    raw = 1.0 - tension / 100.0
    return float(np.clip((raw ** 0.55) * 100, 0, 100))


def _cog_speed_score(n_valid_onsets: int, duration_sec: float,
                     expected_syllables: int) -> float:
    """
    Phrase-relative cognitive speed.

    If expected_syllables is known (phrase was provided):
      completion = n_valid_onsets / expected_syllables
      Score scales from 0 (said nothing) to ~85 (completed phrase at
      normal pace) to 95 (fast and complete).

    If no phrase provided:
      Use absolute onset count with hard floors.
    """
    if expected_syllables > 0:
        # Phrase-relative mode
        completion = n_valid_onsets / expected_syllables
        if completion <= 0:
            return 5.0
        elif completion < 0.15:
            return 5 + completion * 100  # 5-20
        elif completion < 0.40:
            return 20 + (completion - 0.15) * 120  # 20-50
        elif completion < 0.70:
            return 50 + (completion - 0.40) * 100  # 50-80
        elif completion < 1.0:
            return 80 + (completion - 0.70) * 33   # 80-90
        else:
            # Completed or exceeded — great
            rate = n_valid_onsets / duration_sec if duration_sec > 0.5 else 0
            bonus = min(5, rate - 4.0) if rate > 4.0 else 0  # fast-speech bonus
            return float(np.clip(90 + bonus, 90, 97))
    else:
        # Fallback: absolute onset count
        if n_valid_onsets <= 1:
            return 5.0
        elif n_valid_onsets <= 3:
            return 10 + n_valid_onsets * 5  # 15-25
        elif n_valid_onsets <= 6:
            return 25 + (n_valid_onsets - 3) * 10  # 35-55
        else:
            rate = n_valid_onsets / duration_sec if duration_sec > 0.5 else 0
            if rate < 2.0:
                return 50.0
            elif rate < 3.5:
                return 50 + (rate - 2.0) * 15  # 50-72.5
            elif rate <= 5.5:
                return 72.5 + (rate - 3.5) * 7.5  # 72.5-87.5
            else:
                return float(np.clip(87.5 + np.log1p(rate - 5.5) * 3, 87.5, 95))


# ---------------------------------------------------------------------------
# Voice Purifier
# ---------------------------------------------------------------------------

def _purify(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns BOTH the pre-emphasised signal (for onset detection) and
    the original trimmed signal (for spectral tilt and pitch analysis).
    """
    # Trim on original first (before pre-emphasis which changes levels)
    for top_db in (25, 35, 45):
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        if len(yt) / sr >= MIN_DURATION_SEC:
            break
    else:
        if len(yt) / sr < 0.3:
            raise ValueError(f"Audio too short ({len(yt)/sr:.2f}s)")

    y_original = yt.copy()

    # Pre-emphasis on the trimmed signal
    y_emph = np.append(yt[0], yt[1:] - PRE_EMPHASIS_COEFF * yt[:-1])

    return y_emph, y_original


# ---------------------------------------------------------------------------
# CPP
# ---------------------------------------------------------------------------

def _compute_cpp(y: np.ndarray, sr: int) -> float:
    n_fft = 4096
    hop = 512
    cpps = []
    for start in range(0, len(y) - n_fft, hop):
        frame = y[start:start + n_fft] * np.hanning(n_fft)
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        log_spec = np.log10(spectrum + 1e-20)
        cepstrum = np.fft.irfft(log_spec)

        q_min = int(sr / 500)
        q_max = min(int(sr / 60), len(cepstrum) - 1)
        if q_min >= q_max:
            continue

        region = cepstrum[q_min:q_max]
        qs = np.arange(q_min, q_max)
        peak_idx = np.argmax(region)
        peak_val = region[peak_idx]
        coeffs = np.polyfit(qs, region, 1)
        regression_at_peak = np.polyval(coeffs, qs[peak_idx])
        cpps.append(float(peak_val - regression_at_peak))

    return float(np.median(cpps)) if cpps else 0.0


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def _extract_features(y_emph: np.ndarray, y_orig: np.ndarray,
                      sr: int, gender: str) -> dict:
    """
    y_emph:  pre-emphasised signal (for onset detection)
    y_orig:  original signal (for pitch analysis and spectral tilt)
    """
    features: dict = {}
    n_fft = 2048
    hop = 512

    # ── 1. CPP (on original — pre-emphasis distorts cepstral peaks) ────
    cpp = _compute_cpp(y_orig, sr)
    features["cpp"] = round(cpp, 4)

    CPP_HEALTHY = 0.08
    CPP_SEVERE = 0.015
    if cpp >= CPP_HEALTHY:
        cpp_deficit = 0.0
    elif cpp <= CPP_SEVERE:
        cpp_deficit = 1.0
    else:
        cpp_deficit = (CPP_HEALTHY - cpp) / (CPP_HEALTHY - CPP_SEVERE)
    features["cpp_deficit"] = round(float(cpp_deficit), 4)

    # ── 2. F0 Analysis (on original — pre-emphasis shifts pitch) ───────
    # Use fmin=40 to catch very deep growls
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y_orig, fmin=40, fmax=500, sr=sr, hop_length=hop, fill_na=0.0
    )

    voiced_f0 = f0[voiced_flag]
    n_total = len(f0)
    n_voiced = len(voiced_f0)
    creaky_thresh = F0_CREAKY.get(gender, F0_CREAKY["other"])

    if n_voiced >= 3:
        f0_median = float(np.median(voiced_f0))
        n_creaky = int(np.sum(voiced_f0 < creaky_thresh))
        creaky_ratio = n_creaky / n_voiced

        # F0 depth: how far below normal
        f0_range = F0_NORMAL_RANGE.get(gender, F0_NORMAL_RANGE["other"])
        f0_mid = (f0_range[0] + f0_range[1]) / 2
        if f0_median >= f0_mid:
            f0_depth = 0.0
        elif f0_median <= f0_range[0] * 0.6:
            f0_depth = 1.0
        else:
            f0_depth = float(np.clip(
                (f0_mid - f0_median) / (f0_mid - f0_range[0] * 0.6), 0, 1
            ))

        # "Healthy voice" bonus: if PYIN found good pitch in normal range,
        # this indicates the voice is NOT strained. Reduce strain contribution.
        healthy_ratio = 1.0 - creaky_ratio  # % of frames with healthy pitch
    else:
        # PYIN total failure: no pitch detected at all
        # This IS the most severe case — maximise strain indicators
        f0_median = 0.0
        creaky_ratio = 1.0   # was 0.8, now 1.0 — total failure = maximum
        f0_depth = 1.0       # was 0.7, now 1.0
        healthy_ratio = 0.0

    features["f0_median"] = round(f0_median, 1)
    features["n_voiced_frames"] = n_voiced
    features["n_total_frames"] = n_total
    features["creaky_ratio"] = round(float(creaky_ratio), 4)
    features["f0_depth"] = round(float(f0_depth), 4)
    features["healthy_ratio"] = round(float(healthy_ratio), 4)

    unvoiced_ratio = 1.0 - (n_voiced / (n_total + 1e-10))
    # Penalty only for extreme unvoicing (>55%)
    unvoiced_penalty = float(np.clip((unvoiced_ratio - 0.55) / 0.30, 0, 1))
    features["unvoiced_ratio"] = round(float(unvoiced_ratio), 4)
    features["unvoiced_penalty"] = round(float(unvoiced_penalty), 4)

    # ── 3. Spectral Tilt (on ORIGINAL — not pre-emphasised!) ──────────
    S_orig = np.abs(librosa.stft(y_orig, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    low_energy = float(np.mean(S_orig[freqs <= 500, :])) + 1e-20
    high_energy = float(np.mean(S_orig[freqs >= 2000, :])) + 1e-20
    spectral_tilt_db = 10 * np.log10(low_energy / high_energy)

    # Now on original signal: normal speech ~5-15 dB, growl ~18-35+ dB
    tilt_strain = float(np.clip((spectral_tilt_db - 10) / 20, 0, 1))
    features["spectral_tilt_db"] = round(spectral_tilt_db, 2)
    features["tilt_strain"] = round(tilt_strain, 4)

    # ── 4. Syllable Onsets (on pre-emphasised, VOICED-gated) ──────────
    # Compute RMS for energy gating
    rms = librosa.feature.rms(
        y=y_emph, frame_length=n_fft, hop_length=hop
    )[0]
    rms_thresh = np.percentile(rms, 30)

    # Spectral flux on mel spectrogram
    mel_S = librosa.feature.melspectrogram(
        y=y_emph, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=80
    )
    mel_db = librosa.power_to_db(mel_S, ref=np.max)
    flux = np.sum(np.maximum(0, np.diff(mel_db, axis=1)), axis=0)

    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-10
    flux_threshold = med + 2.0 * mad
    flux_abs_floor = np.percentile(flux, 65)
    effective_threshold = max(flux_threshold, flux_abs_floor)

    min_gap = max(1, int(0.13 * sr / hop))

    # Build a voiced-frame mask: True where PYIN detected pitch
    # Expand each voiced frame by ±2 frames to allow for onset timing offset
    voiced_mask = np.zeros(n_total, dtype=bool)
    for i in range(n_total):
        if voiced_flag[i]:
            for j in range(max(0, i - 2), min(n_total, i + 3)):
                voiced_mask[j] = True

    peaks = []
    last = -min_gap - 1
    rms_al = rms[1:len(flux) + 1] if len(rms) > len(flux) else rms[:len(flux)]

    for i in range(1, len(flux) - 1):
        # Energy gate
        if i < len(rms_al) and rms_al[i] < rms_thresh:
            continue
        # Voiced gate: only count onsets near voiced frames
        if i < len(voiced_mask) and not voiced_mask[i]:
            continue
        if (flux[i] > effective_threshold
                and flux[i] >= flux[i - 1]
                and flux[i] >= flux[i + 1]
                and (i - last) >= min_gap):
            peaks.append(i)
            last = i

    duration_sec = len(y_orig) / sr
    n_valid_onsets = len(peaks)
    syllable_rate = n_valid_onsets / duration_sec if duration_sec > 0.1 else 0.0

    features["n_valid_onsets"] = n_valid_onsets
    features["duration_sec"] = round(duration_sec, 2)
    features["syllable_rate"] = round(syllable_rate, 2)

    return features


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_scores(features: dict, is_smoker: bool, gender: str,
                    expected_syllables: int) -> dict:
    """
    Tension weights:
      0.30 * creaky_ratio      — % of voice in fry zone (primary signal)
      0.25 * cpp_deficit       — voice periodicity quality
      0.20 * f0_depth          — abnormally low pitch
      0.15 * tilt_strain       — spectral energy skewed low
      0.10 * unvoiced_penalty  — pitch tracking failures

    HEALTHY VOICE DISCOUNT:
      If PYIN tracked healthy-range pitch for >70% of voiced frames,
      apply a 0.6x multiplier to raw_strain. This ensures that a normal
      voice with good pitch tracking can't score above ~25 tension even
      if one or two other features are borderline.
    """
    cpp_def = features["cpp_deficit"]
    creaky = features["creaky_ratio"]
    f0_depth = features["f0_depth"]
    tilt = features["tilt_strain"]
    unvoiced_pen = features["unvoiced_penalty"]
    healthy = features["healthy_ratio"]
    n_onsets = features["n_valid_onsets"]
    duration = features["duration_sec"]

    raw_strain = (
        0.30 * float(creaky)
        + 0.25 * float(cpp_def)
        + 0.20 * float(f0_depth)
        + 0.15 * float(tilt)
        + 0.10 * float(unvoiced_pen)
    )

    if is_smoker:
        raw_strain += 0.04

    # Healthy voice discount: if most detected pitch was in normal range,
    # the voice is fundamentally healthy — suppress false strain signals
    if healthy >= 0.70:
        discount = 0.5 + 0.5 * (1.0 - healthy)  # 70%→0.65x, 90%→0.55x, 100%→0.50x
        raw_strain *= discount

    raw_strain = float(np.clip(raw_strain, 0, 1.0))

    tension = _tension_curve(raw_strain)
    vitality = _vitality_from_tension(tension)
    cog_speed = _cog_speed_score(n_onsets, duration, expected_syllables)

    vrs = 0.35 * (100 - tension) + 0.40 * vitality + 0.25 * cog_speed

    result = {
        "vrs": round(float(np.clip(vrs, 0, 100)), 1),
        "tension": round(tension, 1),
        "vitality": round(vitality, 1),
        "cog_speed": round(cog_speed, 1),
        "debug": {
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in features.items()},
            "raw_strain": round(float(raw_strain), 4),
            "healthy_discount_applied": healthy >= 0.70,
            "expected_syllables": expected_syllables,
            "is_smoker": is_smoker,
            "gender": gender,
        },
    }

    # Detailed logging
    log.info("=" * 60)
    log.info("VOCAL BIOMARKER v5 RESULTS")
    log.info("=" * 60)
    log.info("SCORES: VRS=%.1f | Tension=%.1f | Vitality=%.1f | CogSpeed=%.1f",
             result["vrs"], result["tension"], result["vitality"], result["cog_speed"])
    log.info("-" * 40)
    log.info("RAW FEATURES:")
    for k, v in features.items():
        log.info("  %-25s = %s", k, v)
    log.info("STRAIN BREAKDOWN:")
    log.info("  creaky_ratio (w=0.30):  %.4f → %.4f", creaky, 0.30 * creaky)
    log.info("  cpp_deficit  (w=0.25):  %.4f → %.4f", cpp_def, 0.25 * cpp_def)
    log.info("  f0_depth     (w=0.20):  %.4f → %.4f", f0_depth, 0.20 * f0_depth)
    log.info("  tilt_strain  (w=0.15):  %.4f → %.4f", tilt, 0.15 * tilt)
    log.info("  unvoiced_pen (w=0.10):  %.4f → %.4f", unvoiced_pen, 0.10 * unvoiced_pen)
    log.info("  raw_strain (pre-discount): -")
    log.info("  healthy_ratio:           %.4f (discount applied: %s)",
             healthy, healthy >= 0.70)
    log.info("  raw_strain (final):      %.4f", raw_strain)
    log.info("  expected_syllables:      %d", expected_syllables)
    log.info("  n_valid_onsets:          %d", n_onsets)
    log.info("=" * 60)

    return result


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=BiomarkerResult)
async def analyze_voice(
    file: UploadFile = File(...),
    is_smoker: bool = Form(False),
    gender: str = Form("other"),
    phrase: str = Form("", description="The reference phrase shown to the user"),
):
    """
    Multipart/form-data upload.

    Fields:
    - file: audio blob (wav, mp3, m4a, ogg, flac, webm)
    - is_smoker: "true" or "false"
    - gender: "male", "female", or "other"
    - phrase: the text shown on screen for the user to read (optional but
      strongly recommended — enables phrase-relative cog_speed scoring)

    Frontend example:
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        formData.append('is_smoker', isSmoker ? 'true' : 'false');
        formData.append('gender', gender || 'other');
        formData.append('phrase', currentPhrase || '');
        const res = await fetch('/analyze', { method: 'POST', body: formData });
        // Do NOT set Content-Type header
    """
    gender = gender.strip().lower()
    if gender not in ("male", "female", "other"):
        gender = "other"

    # Estimate expected syllables from phrase
    expected_syllables = _estimate_syllable_count(phrase) if phrase.strip() else 0
    log.info("Reference phrase: '%s' → estimated %d syllables",
             phrase[:80] if phrase else "(none)", expected_syllables)

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            y, sr = librosa.load(tmp.name, sr=TARGET_SR, mono=True)
    except Exception as exc:
        log.exception("Decode failed")
        raise HTTPException(400, f"Could not decode audio: {exc}")

    try:
        y_emph, y_orig = _purify(y, sr)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    features = _extract_features(y_emph, y_orig, sr, gender)
    scores = _compute_scores(features, is_smoker, gender, expected_syllables)

    return BiomarkerResult(**scores)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
