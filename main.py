"""
Vocal Biomarker API v4 — Fixing the Inversion & Overcounting Bugs
==================================================================

Bugs found in v3 from real test results
-----------------------------------------
BUG 1 — TENSION INVERSION: Normal voice scored HIGHER tension (73.8)
than growl (69.9). Root cause: F0 irregularity was penalizing normal
prosodic variation. Normal speech has INTENDED pitch variation (questions
go up, statements go down) which produced high F0 CV. A steady low growl
has LESS pitch variation — PYIN tracked a consistent low drone.

FIX: Replace F0 CV with F0 FLOOR RATIO — the percentage of voiced frames
where F0 drops below a "creaky threshold" (gender-adjusted). Normal
speech occasionally dips low but mostly stays above; growl lives below
the threshold almost entirely.

BUG 2 — COG SPEED 56.8 FOR ONE SYLLABLE: The spectral flux onset
detector was counting the growl's pulsing, breath noise, and iPhone
mic artifacts as syllable onsets.

FIX: Three changes:
  a) Energy-gate the flux: only count onsets in frames where the RMS
     energy is above 25th percentile (filters out noise/breath).
  b) Require a minimum flux MAGNITUDE (not just "above median+MAD")
     to count as a real onset.
  c) Much harsher completion penalty: if total onsets < 3, clamp
     cog_speed to 0-25 regardless of rate math.

BUG 3 — CPP THRESHOLD MISCALIBRATED: The cepstral peak prominence
calculation was likely giving similar values for both voices because
the regression window was too wide. Tightened the calculation.

NEW DIAGNOSTIC: All raw features are logged at INFO level so you can
see exactly what's happening. Check your Railway/Render logs.
"""

from __future__ import annotations

import logging
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

app = FastAPI(title="Vocal Biomarker API", version="4.0.0")
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

# F0 "creaky" thresholds — below this F0, voice is in the fry/growl zone.
# These are NOT "average" F0 — they're the floor of normal phonation.
F0_CREAKY = {
    "male":   90,    # male vocal fry typically <90 Hz
    "female": 135,   # female vocal fry typically <135 Hz
    "other":  110,
}

# F0 ranges for "normal" median pitch
F0_NORMAL_RANGE = {
    "male":   (85, 180),
    "female": (150, 300),
    "other":  (100, 240),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float, mid: float, slope: float) -> float:
    z = np.clip(slope * (x - mid), -500, 500)
    return float(1.0 / (1.0 + np.exp(-z)))


def _tension_curve(raw_strain: float) -> float:
    """
    raw_strain 0→1+ maps to tension 0→100.

    Design:
      0.00-0.12 (normal)  → tension  2-12
      0.15-0.25 (mild)    → tension 15-30
      0.35-0.50 (moderate)→ tension 45-65
      0.55-0.75 (severe)  → tension 72-90
      0.80+     (extreme) → tension 92-99

    Sigmoid at midpoint=0.38, slope=9.
    """
    s = _sigmoid(raw_strain, mid=0.38, slope=9.0)
    return float(np.clip(s * 100, 0, 100))


def _vitality_from_tension(tension: float) -> float:
    """
    Asymmetric inverse:
      tension  0-15  → vitality 90-100
      tension 15-40  → vitality 60-90
      tension 50-70  → vitality 30-55
      tension 80-100 → vitality  0-20
    """
    raw = 1.0 - tension / 100.0
    return float(np.clip((raw ** 0.55) * 100, 0, 100))


def _cog_speed_score(n_onsets: int, syllable_rate: float,
                     duration_sec: float) -> float:
    """
    Cog speed from onset count + rate, with hard floors.

    Key rule: if fewer than 3 onsets detected, the person barely spoke.
    Cap cog_speed at 20 regardless of what the rate math says.
    """
    # Hard floor: barely any syllables detected
    if n_onsets <= 1:
        return 5.0
    if n_onsets <= 2:
        return 15.0
    if n_onsets <= 4:
        # Some speech but very little
        base = 20 + (n_onsets - 2) * 8  # 28-36
        return float(np.clip(base, 0, 40))

    # Normal onset count (5+): use syllable rate
    if syllable_rate < 1.0:
        return 25.0
    elif syllable_rate < 2.5:
        return 25 + (syllable_rate - 1.0) * 25  # 25-62.5
    elif syllable_rate < 3.5:
        return 62.5 + (syllable_rate - 2.5) * 15  # 62.5-77.5
    elif syllable_rate <= 5.5:
        return 77.5 + (syllable_rate - 3.5) * 5  # 77.5-87.5
    else:
        # Very fast — gentle cap
        return float(np.clip(87.5 + np.log1p(syllable_rate - 5.5) * 3, 0, 95))


# ---------------------------------------------------------------------------
# Voice Purifier
# ---------------------------------------------------------------------------

def _purify(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.append(y[0], y[1:] - PRE_EMPHASIS_COEFF * y[:-1])
    for top_db in (25, 35, 45):
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        if len(yt) / sr >= MIN_DURATION_SEC:
            return yt
    if len(yt) / sr < 0.3:
        raise ValueError(f"Audio too short ({len(yt)/sr:.2f}s)")
    return yt


# ---------------------------------------------------------------------------
# CPP (Cepstral Peak Prominence)
# ---------------------------------------------------------------------------

def _compute_cpp(y: np.ndarray, sr: int) -> float:
    """
    Frame-by-frame CPP, median across frames.
    Uses a 4096-sample window for good quefrency resolution.
    """
    n_fft = 4096
    hop = 512
    cpps = []

    for start in range(0, len(y) - n_fft, hop):
        frame = y[start:start + n_fft] * np.hanning(n_fft)
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        log_spec = np.log10(spectrum + 1e-20)
        cepstrum = np.fft.irfft(log_spec)

        # Quefrency range: 60-500 Hz
        q_min = int(sr / 500)
        q_max = min(int(sr / 60), len(cepstrum) - 1)
        if q_min >= q_max:
            continue

        region = cepstrum[q_min:q_max]
        qs = np.arange(q_min, q_max)

        peak_idx = np.argmax(region)
        peak_val = region[peak_idx]

        # Linear regression through the cepstral region
        coeffs = np.polyfit(qs, region, 1)
        regression_at_peak = np.polyval(coeffs, qs[peak_idx])

        cpps.append(float(peak_val - regression_at_peak))

    return float(np.median(cpps)) if cpps else 0.0


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def _extract_features(y: np.ndarray, sr: int, gender: str) -> dict:
    """
    Four strain features + syllable counting.

    1. CPP deficit        — voice periodicity quality (clinical standard)
    2. F0 creaky ratio    — % of voiced frames below creaky threshold
    3. F0 median depth    — how low is median F0 vs normal range
    4. Spectral tilt      — low-to-high energy ratio (survives AGC)
    5. Syllable onsets    — energy-gated spectral flux peak detection
    """
    features: dict = {}
    n_fft = 2048
    hop = 512

    # ── 1. CPP ──────────────────────────────────────────────────────────
    cpp = _compute_cpp(y, sr)
    features["cpp"] = round(cpp, 4)

    # CPP deficit: healthy ≥ 0.08, severe ≤ 0.02
    # (Note: our CPP is in cepstral units, not clinical dB-CPP.
    #  Scale is much smaller — typically 0.01-0.15 range)
    # We'll normalise based on what we actually observe:
    # Use a relative scale: deficit grows as CPP drops below 0.07
    CPP_HEALTHY = 0.08
    CPP_SEVERE = 0.02
    if cpp >= CPP_HEALTHY:
        cpp_deficit = 0.0
    elif cpp <= CPP_SEVERE:
        cpp_deficit = 1.0
    else:
        cpp_deficit = (CPP_HEALTHY - cpp) / (CPP_HEALTHY - CPP_SEVERE)
    features["cpp_deficit"] = round(float(cpp_deficit), 4)

    # ── 2 & 3. F0 Analysis ─────────────────────────────────────────────
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr, hop_length=hop, fill_na=0.0
    )

    voiced_f0 = f0[voiced_flag]
    n_total = len(f0)
    n_voiced = len(voiced_f0)

    # F0 CREAKY RATIO: % of VOICED frames below the creaky threshold.
    # This is the key fix — we don't care about F0 VARIATION (which
    # punishes normal prosody). We care about how much time the voice
    # spends in the "creaky/fry zone."
    # Normal speech: maybe 0-10% of frames dip that low.
    # Growl/fry: 60-100% of frames are below threshold.
    creaky_thresh = F0_CREAKY.get(gender, F0_CREAKY["other"])

    if n_voiced > 2:
        f0_median = float(np.median(voiced_f0))
        n_creaky = int(np.sum(voiced_f0 < creaky_thresh))
        creaky_ratio = n_creaky / n_voiced
    else:
        f0_median = 0.0
        creaky_ratio = 0.8  # can't detect pitch → very abnormal

    features["f0_median"] = round(f0_median, 1)
    features["n_voiced_frames"] = n_voiced
    features["n_total_frames"] = n_total
    features["creaky_ratio"] = round(float(creaky_ratio), 4)

    # F0 median depth: how far below normal range midpoint
    f0_range = F0_NORMAL_RANGE.get(gender, F0_NORMAL_RANGE["other"])
    f0_midpoint = (f0_range[0] + f0_range[1]) / 2
    if f0_median > 0:
        # Depth: 0 if at/above midpoint, 1 if at/below lower bound
        if f0_median >= f0_midpoint:
            f0_depth = 0.0
        elif f0_median <= f0_range[0] * 0.7:  # well below normal floor
            f0_depth = 1.0
        else:
            f0_depth = np.clip(
                (f0_midpoint - f0_median) / (f0_midpoint - f0_range[0] * 0.7),
                0, 1
            )
    else:
        f0_depth = 0.7

    features["f0_depth"] = round(float(f0_depth), 4)

    # ── 4. Spectral Tilt ───────────────────────────────────────────────
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    low_energy = float(np.mean(S[freqs <= 500, :])) + 1e-20
    high_energy = float(np.mean(S[freqs >= 2000, :])) + 1e-20
    spectral_tilt_db = 10 * np.log10(low_energy / high_energy)

    # Normal: tilt ~5-15 dB. Growl: ~20-40 dB.
    # Strain: 0 at 12 dB, 1 at 35 dB.
    tilt_strain = np.clip((spectral_tilt_db - 12) / 23, 0, 1)

    features["spectral_tilt_db"] = round(spectral_tilt_db, 2)
    features["tilt_strain"] = round(float(tilt_strain), 4)

    # ── 5. Unvoiced Ratio ──────────────────────────────────────────────
    # Additional signal: growl often causes PYIN to lose tracking entirely
    unvoiced_ratio = 1.0 - (n_voiced / (n_total + 1e-10))
    features["unvoiced_ratio"] = round(float(unvoiced_ratio), 4)

    # Penalty if excessively unvoiced (>50% = abnormal for speech)
    unvoiced_penalty = np.clip((unvoiced_ratio - 0.45) / 0.30, 0, 1)
    features["unvoiced_penalty"] = round(float(unvoiced_penalty), 4)

    # ── 6. Syllable Onsets (Energy-Gated) ──────────────────────────────
    # Compute RMS energy per frame for gating
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    rms_threshold = np.percentile(rms, 25)  # bottom 25% = noise/silence

    mel_S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=80
    )
    mel_db = librosa.power_to_db(mel_S, ref=np.max)
    flux = np.sum(np.maximum(0, np.diff(mel_db, axis=1)), axis=0)

    # Adaptive threshold: higher than before (2.0 * MAD)
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-10
    flux_threshold = med + 2.0 * mad

    # Also require a minimum absolute flux magnitude to avoid counting
    # tiny fluctuations. Use 60th percentile as floor.
    flux_abs_floor = np.percentile(flux, 60)
    effective_threshold = max(flux_threshold, flux_abs_floor)

    # Minimum inter-onset: 130ms (stricter)
    min_gap = max(1, int(0.13 * sr / hop))
    peaks = []
    last = -min_gap - 1

    # Note: flux has one fewer frame than rms, align by taking rms[1:]
    rms_aligned = rms[1:len(flux) + 1] if len(rms) > len(flux) else rms[:len(flux)]

    for i in range(1, len(flux) - 1):
        # Energy gate: skip if this frame's RMS is in the noise floor
        if i < len(rms_aligned) and rms_aligned[i] < rms_threshold:
            continue
        if (flux[i] > effective_threshold
                and flux[i] >= flux[i - 1]
                and flux[i] >= flux[i + 1]
                and (i - last) >= min_gap):
            peaks.append(i)
            last = i

    duration_sec = len(y) / sr
    n_onsets = len(peaks)
    syllable_rate = n_onsets / duration_sec if duration_sec > 0.1 else 0.0

    features["n_onsets"] = n_onsets
    features["duration_sec"] = round(duration_sec, 2)
    features["syllable_rate"] = round(syllable_rate, 2)

    return features


# ---------------------------------------------------------------------------
# Composite Scoring
# ---------------------------------------------------------------------------

def _compute_scores(features: dict, is_smoker: bool, gender: str) -> dict:
    """
    Tension from 4 strain components:

      0.25 * cpp_deficit       — voice periodicity (clinical gold standard)
      0.30 * creaky_ratio      — % of time in vocal fry zone (KEY FIX)
      0.20 * f0_depth          — how abnormally low the median pitch is
      0.15 * tilt_strain       — spectral energy skewed to low frequencies
      0.10 * unvoiced_penalty  — pitch tracking failures (severe cases)

    Why CREAKY RATIO is the primary signal (weight 0.30):
    - It directly measures "is this voice in the fry/growl register?"
    - Normal speech: 0-10% of frames below creaky threshold → near 0
    - Deep growl: 70-100% of frames below threshold → near 1
    - It CANNOT be confused by prosodic variation (the v3 bug)
    - It survives iPhone AGC because it's frequency-based, not level-based
    """
    cpp_def = features["cpp_deficit"]
    creaky = features["creaky_ratio"]
    f0_depth = features["f0_depth"]
    tilt = features["tilt_strain"]
    unvoiced_pen = features["unvoiced_penalty"]
    n_onsets = features["n_onsets"]
    syl_rate = features["syllable_rate"]
    duration = features["duration_sec"]

    raw_strain = (
        0.25 * float(cpp_def)
        + 0.30 * float(creaky)
        + 0.20 * float(f0_depth)
        + 0.15 * float(tilt)
        + 0.10 * float(unvoiced_pen)
    )

    if is_smoker:
        raw_strain += 0.04

    raw_strain = float(np.clip(raw_strain, 0, 1.0))

    tension = _tension_curve(raw_strain)
    vitality = _vitality_from_tension(tension)
    cog_speed = _cog_speed_score(n_onsets, syl_rate, duration)

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
            "is_smoker": is_smoker,
            "gender": gender,
        },
    }

    # Log everything for diagnostics
    log.info("=" * 60)
    log.info("VOCAL BIOMARKER ANALYSIS RESULTS")
    log.info("=" * 60)
    log.info("SCORES: VRS=%.1f | Tension=%.1f | Vitality=%.1f | CogSpeed=%.1f",
             result["vrs"], result["tension"], result["vitality"], result["cog_speed"])
    log.info("-" * 40)
    log.info("RAW FEATURES:")
    for k, v in features.items():
        log.info("  %-25s = %s", k, v)
    log.info("  %-25s = %.4f", "raw_strain", raw_strain)
    log.info("STRAIN COMPONENTS:")
    log.info("  cpp_deficit  (w=0.25): %.4f → contribution %.4f", cpp_def, 0.25 * cpp_def)
    log.info("  creaky_ratio (w=0.30): %.4f → contribution %.4f", creaky, 0.30 * creaky)
    log.info("  f0_depth     (w=0.20): %.4f → contribution %.4f", f0_depth, 0.20 * f0_depth)
    log.info("  tilt_strain  (w=0.15): %.4f → contribution %.4f", tilt, 0.15 * tilt)
    log.info("  unvoiced_pen (w=0.10): %.4f → contribution %.4f", unvoiced_pen, 0.10 * unvoiced_pen)
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
):
    """
    Multipart/form-data upload. Do NOT set Content-Type header manually.

    Frontend example:
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        formData.append('is_smoker', isSmoker ? 'true' : 'false');
        formData.append('gender', gender || 'other');
        const res = await fetch('/analyze', { method: 'POST', body: formData });
    """
    gender = gender.strip().lower()
    if gender not in ("male", "female", "other"):
        gender = "other"

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
        y = _purify(y, sr)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    features = _extract_features(y, sr, gender)
    scores = _compute_scores(features, is_smoker, gender)

    return BiomarkerResult(**scores)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
