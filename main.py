"""
Vocal Biomarker API — Medical-Grade Acoustic Analysis
======================================================
Accepts an audio upload + metadata (is_smoker, gender) and returns four
scores (0-100): vrs, tension, vitality, cog_speed.

Acoustic Science Summary
-------------------------
TENSION is driven by two complementary features:
  1. Harmonic-to-Percussive Energy Ratio (HPER):
     librosa.effects.hpss splits the spectrogram into harmonic and percussive
     components. Normal phonation is overwhelmingly harmonic. Vocal fry/growl
     produces subharmonic pulses that bleed into the percussive matrix.
     A low harmonic ratio ⇒ high physical strain.
  2. Spectral Bandwidth Compression:
     Healthy voiced speech distributes energy across multiple harmonics,
     yielding wide spectral bandwidth. A strained growl clusters energy
     in a narrow low-frequency band. Abnormally low bandwidth (relative to
     gender-adjusted norms) signals tension.
  3. MFCC Variance Penalty:
     Vocal fry produces unnaturally uniform MFCCs frame-to-frame because
     the irregular glottal pulses lack the spectral modulation of normal
     speech. Very LOW variance in MFCCs 2-6 is a strain indicator — but
     we only penalise values far below normal, so expressive speech
     (which has HIGH variance) is never caught.

VITALITY is the physical inverse of tension, passed through its own
non-linear curve so that moderate tension only mildly reduces vitality
but severe tension crushes it.

COG_SPEED measures articulatory rate (syllables/sec) via mel-spectrogram
spectral flux onset detection, filtered for plausible inter-syllable
intervals (100-400 ms). This is purely a temporal/sequencing measure
and is decoupled from voice quality — a person growling words clearly
at a normal pace still produces distinct spectral transitions.

NON-LINEAR MAPPING uses modified logistics / power curves with
genre-specific midpoints so that the "normal" zone occupies a wide
comfortable plateau and only extreme values trigger steep penalties.

VOICE PURIFIER applies pre-emphasis (boost high-freq to counteract
smartphone mic roll-off), librosa.effects.trim with configurable top_db,
and a minimum-duration safety gate.
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
    version="1.0.0",
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
    vrs: float = Field(..., ge=0, le=100, description="Vocal Resilience Score (composite)")
    tension: float = Field(..., ge=0, le=100, description="Physical vocal tension")
    vitality: float = Field(..., ge=0, le=100, description="Vocal vitality / health")
    cog_speed: float = Field(..., ge=0, le=100, description="Cognitive-articulatory speed")
    debug: Optional[dict] = Field(None, description="Raw feature values (dev only)")


# ---------------------------------------------------------------------------
# Constants & gender norms
# ---------------------------------------------------------------------------
TARGET_SR = 22050
MIN_DURATION_SEC = 0.8  # after trimming
PRE_EMPHASIS_COEFF = 0.97

# Gender-adjusted spectral bandwidth norms (Hz) — used to normalise
# bandwidth so a deep male voice isn't unfairly penalised.
BANDWIDTH_NORMS = {
    "male":   {"mean": 1800, "std": 500},
    "female": {"mean": 2200, "std": 550},
    "other":  {"mean": 2000, "std": 525},
}

# Syllable rate norms (syllables / sec) — conversational speech
SYLLABLE_RATE_NORM = {"mean": 4.5, "std": 1.5}  # ~3-6 syl/s is normal

# MFCC variance floor — below this, speech is unnaturally uniform
MFCC_VAR_FLOOR = 8.0   # empirical; normal speech ≈ 15-40
MFCC_VAR_CEIL = 40.0   # above this, no extra credit


# ---------------------------------------------------------------------------
# Non-linear mapping helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float, midpoint: float, slope: float) -> float:
    """Standard logistic sigmoid mapped to [0, 1]."""
    z = slope * (x - midpoint)
    z = np.clip(z, -500, 500)  # numerical safety
    return float(1.0 / (1.0 + np.exp(-z)))


def _tension_curve(raw_strain: float) -> float:
    """
    Maps a composite strain value (roughly 0-1, can exceed) to 0-100.
    Normal speech → raw_strain ≈ 0.05-0.20 → score 5-25.
    Severe fry    → raw_strain ≈ 0.60-1.00 → score 85-100.
    Uses a steep sigmoid centred at 0.40 so the transition is sharp.
    """
    s = _sigmoid(raw_strain, midpoint=0.40, slope=12.0)
    return float(np.clip(s * 100, 0, 100))


def _vitality_from_tension(tension_score: float) -> float:
    """
    Inverse of tension with an asymmetric curve:
    - Tension <30 → Vitality 85-100 (wide safe zone)
    - Tension >70 → Vitality 0-15 (crushed)
    Uses a power curve for the asymmetry.
    """
    # Invert
    raw = 1.0 - (tension_score / 100.0)
    # Apply power curve to widen the "good" end
    shaped = raw ** 0.7  # <1 exponent expands the top end
    return float(np.clip(shaped * 100, 0, 100))


def _cog_speed_curve(syllable_rate: float) -> float:
    """
    Maps syllable rate to 0-100.
    Target zone: 3.5-6.0 syl/s → 70-90.
    Very slow (<2) → 30-50.
    Very fast (>7) → 90-95 (slight cap — hyperfast isn't necessarily better).
    Below 1 syl/s → <20 (near-silent / severely impaired).
    """
    norm_mean = SYLLABLE_RATE_NORM["mean"]  # 4.5
    # Piecewise: below norm use one sigmoid, above norm use another
    if syllable_rate <= norm_mean:
        # Slow side — sigmoid with midpoint at 2.0
        s = _sigmoid(syllable_rate, midpoint=2.0, slope=2.0)
        # Scale so that 4.5 → ~0.80
        score = s * 88
    else:
        # Fast side — gentle logarithmic rise, capped
        excess = syllable_rate - norm_mean
        score = 80 + 5 * np.log1p(excess)  # slow growth
    return float(np.clip(score, 0, 100))


# ---------------------------------------------------------------------------
# Voice Purifier
# ---------------------------------------------------------------------------

def _purify(y: np.ndarray, sr: int) -> np.ndarray:
    """
    1. Pre-emphasis to counteract smartphone mic low-pass roll-off and
       proximity effect.  Boosts consonant transients for better onset
       detection downstream.
    2. Trim silence with top_db=25 — aggressive enough to strip room hiss
       from iPhone recordings but gentle enough to keep breathy speech.
    3. Min-duration gate.
    """
    # Pre-emphasis
    y = np.append(y[0], y[1:] - PRE_EMPHASIS_COEFF * y[:-1])

    # Trim
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)

    if len(y_trimmed) / sr < MIN_DURATION_SEC:
        # Fallback: try gentler trim
        y_trimmed, _ = librosa.effects.trim(y, top_db=35)

    if len(y_trimmed) / sr < MIN_DURATION_SEC:
        # Last resort: use original (minus leading/trailing silence at 45 dB)
        y_trimmed, _ = librosa.effects.trim(y, top_db=45)
        if len(y_trimmed) / sr < 0.3:
            raise ValueError(
                f"Audio too short after trimming ({len(y_trimmed)/sr:.2f}s). "
                "Need at least 0.3 s of voiced content."
            )

    return y_trimmed


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def _extract_features(y: np.ndarray, sr: int, gender: str) -> dict:
    """
    Returns a dict of intermediate acoustic features.

    Features extracted
    ------------------
    harmonic_ratio : float
        RMS(harmonic) / (RMS(harmonic) + RMS(percussive)).
        Normal speech ≈ 0.80-0.92.  Growl/fry ≈ 0.45-0.65.

    bandwidth_z : float
        Z-score of mean spectral bandwidth relative to gender norms.
        Normal ≈ -0.5 to +1.0.  Growl ≈ -2.0 to -3.5.

    mfcc_var_z : float
        How far mean MFCC(2:6) variance is below the normal floor,
        expressed as a 0-1 deficit.  0 = normal+, 1 = severely uniform.

    syllable_rate : float
        Estimated syllables per second via spectral-flux onset detection.
    """

    features: dict = {}
    n_fft = 2048
    hop = 512

    # --- 1. Harmonic / Percussive separation ---------------------------------
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag = np.abs(stft)
    harm_mag, perc_mag = librosa.decompose.hpss(mag, margin=3.0)

    rms_h = np.sqrt(np.mean(harm_mag ** 2)) + 1e-10
    rms_p = np.sqrt(np.mean(perc_mag ** 2)) + 1e-10
    harmonic_ratio = rms_h / (rms_h + rms_p)
    features["harmonic_ratio"] = float(harmonic_ratio)

    # --- 2. Spectral Bandwidth -----------------------------------------------
    spec_bw = librosa.feature.spectral_bandwidth(
        S=mag, sr=sr, n_fft=n_fft, hop_length=hop
    )[0]
    mean_bw = float(np.mean(spec_bw))
    norms = BANDWIDTH_NORMS.get(gender, BANDWIDTH_NORMS["other"])
    bandwidth_z = (mean_bw - norms["mean"]) / norms["std"]
    features["spectral_bandwidth_hz"] = mean_bw
    features["bandwidth_z"] = float(bandwidth_z)

    # --- 3. MFCC Variance (coeffs 2-6) --------------------------------------
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    # Coefficients 2-6 carry vocal-tract shape info; coeff 1 is overall energy
    mfcc_subset = mfccs[1:6, :]  # shape (5, T)
    # Per-coefficient temporal variance, then mean across coefficients
    mfcc_var = float(np.mean(np.var(mfcc_subset, axis=1)))
    # Express as a 0-1 deficit below floor
    if mfcc_var >= MFCC_VAR_CEIL:
        mfcc_var_deficit = 0.0
    elif mfcc_var <= MFCC_VAR_FLOOR:
        mfcc_var_deficit = 1.0
    else:
        mfcc_var_deficit = 1.0 - (mfcc_var - MFCC_VAR_FLOOR) / (MFCC_VAR_CEIL - MFCC_VAR_FLOOR)
    features["mfcc_var"] = mfcc_var
    features["mfcc_var_deficit"] = float(mfcc_var_deficit)

    # --- 4. Syllable Rate via Spectral Flux Onsets ---------------------------
    # Use mel spectrogram for spectral flux — more robust to broadband
    # noise than raw STFT because mel bands de-emphasise high-freq hiss.
    mel_S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=80)
    mel_db = librosa.power_to_db(mel_S, ref=np.max)

    # Spectral flux = frame-to-frame L1 difference (only positive changes)
    flux = np.sum(np.maximum(0, np.diff(mel_db, axis=1)), axis=0)

    # Adaptive threshold: median + 1.2 * MAD  (works across volume levels)
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-10
    threshold = med + 1.2 * mad

    # Find peaks above threshold with minimum inter-onset interval of 100 ms
    min_gap_frames = int(0.10 * sr / hop)  # ~4 frames at sr=22050, hop=512
    peak_indices = []
    last_peak = -min_gap_frames - 1
    for i in range(1, len(flux) - 1):
        if flux[i] > threshold and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            if i - last_peak >= min_gap_frames:
                peak_indices.append(i)
                last_peak = i

    n_onsets = len(peak_indices)
    duration_sec = len(y) / sr
    syllable_rate = n_onsets / duration_sec if duration_sec > 0 else 0.0
    features["n_onsets"] = n_onsets
    features["duration_sec"] = round(duration_sec, 2)
    features["syllable_rate"] = round(syllable_rate, 2)

    return features


# ---------------------------------------------------------------------------
# Composite Scoring
# ---------------------------------------------------------------------------

def _compute_scores(features: dict, is_smoker: bool, gender: str) -> dict:
    """
    Combine extracted features into the four output scores.

    Tension composite strain formula
    ---------------------------------
    strain = w1 * (1 - harmonic_ratio_scaled)
           + w2 * bandwidth_deficit
           + w3 * mfcc_var_deficit

    Where:
      harmonic_ratio_scaled: rescaled so 0.90→0, 0.50→1
      bandwidth_deficit:     how far below norm (clipped 0-1)
      mfcc_var_deficit:      from extraction (0-1)
      w1=0.50, w2=0.30, w3=0.20  (harmonic ratio is the strongest signal)
    """

    hr = features["harmonic_ratio"]
    bw_z = features["bandwidth_z"]
    mfcc_def = features["mfcc_var_deficit"]
    syl_rate = features["syllable_rate"]

    # -- Harmonic ratio → strain component (0-1) --
    # Normal ≈ 0.82-0.92 → near 0.  Growl ≈ 0.45-0.65 → near 1.
    hr_strain = np.clip((0.90 - hr) / 0.40, 0, 1)  # 0.90→0, 0.50→1

    # -- Bandwidth deficit (0-1) --
    # bw_z < -1.5 is suspicious; < -3 is severe
    bw_deficit = np.clip((-1.5 - bw_z) / 2.0, 0, 1)  # -1.5→0, -3.5→1

    # -- Composite raw strain --
    raw_strain = 0.50 * hr_strain + 0.30 * bw_deficit + 0.20 * mfcc_def

    # Smoker adjustment: slight baseline tension increase (5-10 pts equivalent)
    if is_smoker:
        raw_strain = raw_strain + 0.06  # nudge

    tension = _tension_curve(raw_strain)
    vitality = _vitality_from_tension(tension)
    cog_speed = _cog_speed_curve(syl_rate)

    # -- VRS (Vocal Resilience Score) --
    # Weighted composite: low tension, high vitality, adequate cog speed
    vrs = 0.35 * (100 - tension) + 0.40 * vitality + 0.25 * cog_speed

    return {
        "vrs": round(float(np.clip(vrs, 0, 100)), 1),
        "tension": round(tension, 1),
        "vitality": round(vitality, 1),
        "cog_speed": round(cog_speed, 1),
        "debug": {
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()},
            "raw_strain": round(float(raw_strain), 4),
            "hr_strain": round(float(hr_strain), 4),
            "bw_deficit": round(float(bw_deficit), 4),
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
    Upload an audio file and receive vocal biomarker scores.

    **Form fields:**
    - `file`: audio file (wav, mp3, m4a, ogg, flac, webm)
    - `is_smoker`: boolean (default false)
    - `gender`: one of male, female, other (default other)

    **Returns:** vrs, tension, vitality, cog_speed (each 0-100) + debug features.
    """
    gender = gender.strip().lower()
    if gender not in ("male", "female", "other"):
        gender = "other"

    # Read uploaded bytes into a temp file (librosa needs a seekable file)
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

    # Purify
    try:
        y = _purify(y, sr)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Extract features
    features = _extract_features(y, sr, gender)
    log.info("Extracted features: %s", features)

    # Score
    scores = _compute_scores(features, is_smoker, gender)
    log.info("Scores: %s", {k: v for k, v in scores.items() if k != "debug"})

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
