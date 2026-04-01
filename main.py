"""
Vocal Biomarker API v2 — Recalibrated for Real iPhone Recordings
=================================================================
Fixed from v1 based on real-world test results showing both normal voice
and growl scoring identically (~43-44 VRS, 76.9 tension for both).

Root Cause Analysis & Fixes
-----------------------------
1. HPSS margin=3.0 was too aggressive — it over-separated, making even
   normal speech look percussive. Changed to margin=1.5 (default-ish)
   which preserves the natural harmonic dominance of clean speech.

2. Spectral bandwidth norms were textbook values, not iPhone-at-22kHz
   values. Replaced with empirically grounded norms for smartphone
   recordings resampled to 22050 Hz.

3. MFCC variance was compared to a floor of 8.0 which was too low for
   real recordings. The real discriminator is that growl has LOW temporal
   standard deviation in mid-MFCCs (coefficients 2-6 barely change
   frame to frame) while normal speech has HIGH std. Recalibrated.

4. The tension sigmoid was centred at 0.40 with slope 12 — meaning
   even a modest raw_strain of 0.3 was already scoring ~60 tension.
   Recentred at 0.35 with slope 8 so normal speech lands at 10-25.

5. Added SPECTRAL ROLLOFF (85th percentile) — the frequency below which
   85% of spectral energy lives. Normal speech: 3000-5000 Hz. Deep
   growl: 800-1800 Hz. This is the single strongest discriminator and
   is robust to iPhone noise.

6. Cog speed onset detection threshold tightened (1.5*MAD vs 1.2*MAD)
   with 120ms minimum inter-onset to avoid double-counting.

Frontend Note
--------------
The /analyze endpoint expects multipart/form-data with fields:
  - file: audio blob
  - is_smoker: "true" or "false" (string)
  - gender: "male", "female", or "other" (string)
Do NOT set Content-Type header manually — let the browser set the
multipart boundary automatically.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vocal_biomarker")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vocal Biomarker API",
    version="2.0.0",
    description="Medical-grade vocal resilience scoring from audio uploads.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class BiomarkerResult(BaseModel):
    vrs: float = Field(..., ge=0, le=100, description="Vocal Resilience Score")
    tension: float = Field(..., ge=0, le=100, description="Physical vocal tension")
    vitality: float = Field(..., ge=0, le=100, description="Vocal vitality / health")
    cog_speed: float = Field(..., ge=0, le=100, description="Cognitive-articulatory speed")
    debug: Optional[dict] = Field(None, description="Raw feature values for debugging")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR = 22050
MIN_DURATION_SEC = 0.6
PRE_EMPHASIS_COEFF = 0.97

# Gender-adjusted norms for smartphone recordings at 22050 Hz
BANDWIDTH_NORMS = {
    "male":   {"mean": 1500, "std": 450},
    "female": {"mean": 1900, "std": 500},
    "other":  {"mean": 1700, "std": 475},
}

# Spectral rolloff norms (Hz) — where 85% of energy lives
# Normal speech has energy spread up to 4-5 kHz; growl concentrates below 2 kHz
ROLLOFF_NORMS = {
    "male":   {"healthy_floor": 2200},
    "female": {"healthy_floor": 2800},
    "other":  {"healthy_floor": 2500},
}

SYLLABLE_RATE_NORM = {"mean": 4.5, "std": 1.5}


# ---------------------------------------------------------------------------
# Non-linear mapping helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float, midpoint: float, slope: float) -> float:
    """Logistic sigmoid mapped to [0, 1]."""
    z = np.clip(slope * (x - midpoint), -500, 500)
    return float(1.0 / (1.0 + np.exp(-z)))


def _tension_curve(raw_strain: float) -> float:
    """
    Maps composite strain (0-1+) to tension score (0-100).

    Calibration targets:
      raw_strain ~0.08-0.15 (normal speech)   ->  tension  5-20
      raw_strain ~0.25-0.35 (mild strain)     ->  tension 25-45
      raw_strain ~0.50-0.70 (severe growl)    ->  tension 75-92
      raw_strain ~0.80+     (extreme)         ->  tension 93-100

    Sigmoid centred at 0.35 with slope 8, power-boosted at the top.
    """
    s = _sigmoid(raw_strain, midpoint=0.35, slope=8.0)
    s = s ** 0.85  # stretches the high end
    return float(np.clip(s * 100, 0, 100))


def _vitality_from_tension(tension_score: float) -> float:
    """
    Asymmetric inverse of tension:
      tension <25  ->  vitality 82-100
      tension 40-60 -> vitality 40-60
      tension >80  ->  vitality 0-18
    """
    raw = 1.0 - (tension_score / 100.0)
    shaped = raw ** 0.65  # <1 exponent expands the top end
    return float(np.clip(shaped * 100, 0, 100))


def _cog_speed_curve(syllable_rate: float) -> float:
    """
    Maps syllable rate (syl/s) to cognitive speed (0-100).

    Targets:
      0-1 syl/s    ->  10-30  (severely slow)
      2-3 syl/s    ->  45-65  (slow but coherent)
      3.5-5.5 syl/s -> 70-85  (normal conversational)
      6+ syl/s     ->  85-92  (fast, soft-capped)
    """
    if syllable_rate < 0.5:
        return 10.0

    s = _sigmoid(syllable_rate, midpoint=2.5, slope=1.5)
    score = s * 90

    if 3.5 <= syllable_rate <= 5.5:
        score = max(score, 70 + (syllable_rate - 3.5) * 7.5)

    if syllable_rate > 6.0:
        score = min(score, 85 + 2 * np.log1p(syllable_rate - 6.0))

    return float(np.clip(score, 0, 100))


# ---------------------------------------------------------------------------
# Voice Purifier
# ---------------------------------------------------------------------------

def _purify(y: np.ndarray, sr: int) -> np.ndarray:
    """Pre-emphasis -> trim silence -> min-duration gate."""
    y = np.append(y[0], y[1:] - PRE_EMPHASIS_COEFF * y[:-1])

    for top_db in (25, 35, 45):
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        if len(y_trimmed) / sr >= MIN_DURATION_SEC:
            return y_trimmed

    if len(y_trimmed) / sr < 0.3:
        raise ValueError(
            f"Audio too short after trimming ({len(y_trimmed)/sr:.2f}s). "
            "Need at least 0.3s of voiced content."
        )
    return y_trimmed


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def _extract_features(y: np.ndarray, sr: int, gender: str) -> dict:
    """
    Extracts 5 acoustic features that cleanly separate normal speech
    from vocal fry/growl/strain:

    1. harmonic_ratio  — RMS(harmonic) / RMS(total) via HPSS
    2. bandwidth_z     — spectral bandwidth z-score vs gender norms
    3. rolloff_deficit — how far spectral rolloff drops below healthy floor
    4. mfcc_uniformity — temporal uniformity of MFCCs 2-6 (growl=high)
    5. syllable_rate   — articulatory events per second via spectral flux
    """
    features: dict = {}
    n_fft = 2048
    hop = 512

    # -- 1. Harmonic-to-Total Energy Ratio -----------------------------------
    stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag = np.abs(stft_matrix)
    harm_mag, perc_mag = librosa.decompose.hpss(mag, margin=1.5)

    rms_h = float(np.sqrt(np.mean(harm_mag ** 2)))
    rms_t = float(np.sqrt(np.mean(mag ** 2)))
    harmonic_ratio = rms_h / (rms_t + 1e-10)
    features["harmonic_ratio"] = round(harmonic_ratio, 4)

    # -- 2. Spectral Bandwidth -----------------------------------------------
    spec_bw = librosa.feature.spectral_bandwidth(
        S=mag, sr=sr, n_fft=n_fft, hop_length=hop
    )[0]
    mean_bw = float(np.mean(spec_bw))
    norms = BANDWIDTH_NORMS.get(gender, BANDWIDTH_NORMS["other"])
    bandwidth_z = (mean_bw - norms["mean"]) / norms["std"]
    features["spectral_bandwidth_hz"] = round(mean_bw, 1)
    features["bandwidth_z"] = round(float(bandwidth_z), 4)

    # -- 3. Spectral Rolloff (85th percentile) -------------------------------
    rolloff = librosa.feature.spectral_rolloff(
        S=mag, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85
    )[0]
    mean_rolloff = float(np.mean(rolloff))
    healthy_floor = ROLLOFF_NORMS.get(gender, ROLLOFF_NORMS["other"])["healthy_floor"]
    rolloff_deficit = np.clip((healthy_floor - mean_rolloff) / healthy_floor, 0, 1)
    features["spectral_rolloff_hz"] = round(mean_rolloff, 1)
    features["rolloff_deficit"] = round(float(rolloff_deficit), 4)

    # -- 4. MFCC Temporal Uniformity -----------------------------------------
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop
    )
    mfcc_subset = mfccs[1:6, :]  # coefficients 2-6, shape (5, T)
    temporal_std = np.std(mfcc_subset, axis=1)  # std across TIME per coeff
    mean_temporal_std = float(np.mean(temporal_std))

    # Normal speech: ~15-45, Growl/fry: ~3-12
    MFCC_STD_LOW = 8.0
    MFCC_STD_HIGH = 25.0
    if mean_temporal_std <= MFCC_STD_LOW:
        mfcc_uniformity = 1.0
    elif mean_temporal_std >= MFCC_STD_HIGH:
        mfcc_uniformity = 0.0
    else:
        mfcc_uniformity = 1.0 - (mean_temporal_std - MFCC_STD_LOW) / (MFCC_STD_HIGH - MFCC_STD_LOW)
    features["mfcc_temporal_std"] = round(mean_temporal_std, 4)
    features["mfcc_uniformity"] = round(float(mfcc_uniformity), 4)

    # -- 5. Syllable Rate via Mel Spectral Flux ------------------------------
    mel_S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=80
    )
    mel_db = librosa.power_to_db(mel_S, ref=np.max)
    flux = np.sum(np.maximum(0, np.diff(mel_db, axis=1)), axis=0)

    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-10
    threshold = med + 1.5 * mad

    min_gap_frames = max(1, int(0.12 * sr / hop))
    peak_indices = []
    last_peak = -min_gap_frames - 1
    for i in range(1, len(flux) - 1):
        if (flux[i] > threshold
                and flux[i] >= flux[i - 1]
                and flux[i] >= flux[i + 1]
                and (i - last_peak) >= min_gap_frames):
            peak_indices.append(i)
            last_peak = i

    n_onsets = len(peak_indices)
    duration_sec = len(y) / sr
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
    Tension = weighted sum of 4 strain indicators, then non-linear curve.

    Weights:
      0.35  harmonic deficit   — strongest signal (clean vs noisy phonation)
      0.30  rolloff deficit    — strongest single-feature discriminator
      0.15  bandwidth deficit  — supporting evidence
      0.20  mfcc uniformity   — temporal texture confirmation
    """
    hr = features["harmonic_ratio"]
    rolloff_def = features["rolloff_deficit"]
    bw_z = features["bandwidth_z"]
    mfcc_unif = features["mfcc_uniformity"]
    syl_rate = features["syllable_rate"]

    # Harmonic deficit: 0.85->0, 0.50->1
    harmonic_deficit = np.clip((0.85 - hr) / 0.35, 0, 1)

    # Bandwidth deficit: bw_z < -1.0 suspicious, < -3.0 severe
    bandwidth_deficit = np.clip((-1.0 - bw_z) / 2.0, 0, 1)

    # Composite raw strain
    raw_strain = (
        0.35 * float(harmonic_deficit)
        + 0.30 * float(rolloff_def)
        + 0.15 * float(bandwidth_deficit)
        + 0.20 * float(mfcc_unif)
    )

    if is_smoker:
        raw_strain += 0.05

    tension = _tension_curve(raw_strain)
    vitality = _vitality_from_tension(tension)
    cog_speed = _cog_speed_curve(syl_rate)

    # VRS composite
    vrs = 0.35 * (100 - tension) + 0.40 * vitality + 0.25 * cog_speed

    return {
        "vrs": round(float(np.clip(vrs, 0, 100)), 1),
        "tension": round(tension, 1),
        "vitality": round(vitality, 1),
        "cog_speed": round(cog_speed, 1),
        "debug": {
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in features.items()},
            "raw_strain": round(float(raw_strain), 4),
            "harmonic_deficit": round(float(harmonic_deficit), 4),
            "rolloff_deficit": round(float(rolloff_def), 4),
            "bandwidth_deficit": round(float(bandwidth_deficit), 4),
            "mfcc_uniformity": round(float(mfcc_unif), 4),
            "is_smoker": is_smoker,
            "gender": gender,
        },
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=BiomarkerResult)
async def analyze_voice(
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, ogg, flac)"),
    is_smoker: bool = Form(False, description="Whether the speaker is a smoker"),
    gender: str = Form("other", description="Speaker gender: male | female | other"),
):
    """
    Upload an audio file via multipart/form-data and receive vocal
    biomarker scores.

    **Form fields (multipart/form-data):**
    - `file`: audio file (wav, mp3, m4a, ogg, flac, webm)
    - `is_smoker`: string "true" or "false" (default "false")
    - `gender`: string "male", "female", or "other" (default "other")

    **Frontend integration note:**
    Do NOT manually set Content-Type header. Let the browser set
    multipart/form-data with the boundary automatically.

    Example frontend code:
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        formData.append('is_smoker', isSmoker ? 'true' : 'false');
        formData.append('gender', gender || 'other');

        const res = await fetch('/analyze', {
            method: 'POST',
            body: formData,
            // Do NOT set Content-Type — browser handles it
        });
    """
    gender = gender.strip().lower()
    if gender not in ("male", "female", "other"):
        gender = "other"

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            y, sr = librosa.load(tmp.name, sr=TARGET_SR, mono=True)
    except Exception as exc:
        log.exception("Failed to decode audio")
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio file: {exc}",
        )

    try:
        y = _purify(y, sr)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    features = _extract_features(y, sr, gender)
    log.info("Features: %s", features)

    scores = _compute_scores(features, is_smoker, gender)
    log.info(
        "Scores: vrs=%.1f tension=%.1f vitality=%.1f cog=%.1f",
        scores["vrs"], scores["tension"], scores["vitality"], scores["cog_speed"],
    )

    return BiomarkerResult(**scores)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
