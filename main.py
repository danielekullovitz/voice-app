"""
Vocal Biomarker API v3 — Rebuilt for iPhone Reality
=====================================================

Why v1 and v2 failed
---------------------
Both versions relied on spectral-domain features (HPSS harmonic ratio,
spectral bandwidth, spectral rolloff) that are neutralised by iPhone
Automatic Gain Control (AGC) and built-in compression. The phone's
signal processing normalises level, boosts mid-range, and compresses
dynamics — making a deep growl look spectrally similar to normal speech
in the processed recording.

v3 Strategy: Use features that survive phone processing
-------------------------------------------------------
iPhone AGC cannot hide these properties:

1. **F0 (Pitch) Statistics** — A deep growl has abnormally LOW fundamental
   frequency (often <90 Hz for males, <140 Hz for females) with HIGH
   irregularity (jitter in F0 track, not waveform jitter). Normal speech
   has higher F0 with smooth prosodic contour.

2. **Cepstral Peak Prominence (CPP)** — The gold standard in clinical
   voice assessment. CPP measures how clearly the voice's fundamental
   period stands out in the cepstrum. Normal phonation: strong, clear
   cepstral peak (high CPP). Vocal fry/growl: weak, smeared peak
   (low CPP) because subharmonic and aperiodic energy disrupt the
   periodicity. CPP is robust to recording conditions because it's a
   RATIO measure.

3. **F0 Contour Regularity** — Normal speech has smooth F0 transitions
   (prosody). Vocal fry produces erratic F0 jumps, dropouts, and
   subharmonic halving. We measure this as the coefficient of variation
   and the percentage of "unvoiced" frames (where pyin can't find F0).

4. **Low-to-High Energy Ratio (Spectral Tilt)** — Even with AGC, the
   RATIO of energy below 500 Hz to energy above 2000 Hz differs
   dramatically: growl concentrates energy low, normal speech spreads it.
   AGC scales the whole spectrum but preserves the ratio.

5. **Syllable Rate with Completion Estimate** — Now estimates what
   fraction of expected speech was actually produced, so an incomplete
   phrase correctly reduces cog_speed.

Frontend Note: multipart/form-data, do NOT set Content-Type manually.
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

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vocal_biomarker")

app = FastAPI(title="Vocal Biomarker API", version="3.0.0")
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

# F0 norms by gender (Hz) — conversational speech
F0_NORMS = {
    "male":   {"mean": 130, "low_threshold": 85,  "very_low": 65},
    "female": {"mean": 210, "low_threshold": 140, "very_low": 100},
    "other":  {"mean": 165, "low_threshold": 110, "very_low": 80},
}

# CPP norms (dB) — from clinical literature
# Normal ≈ 8-14 dB, Dysphonic ≈ 3-7 dB, Severe ≈ <3 dB
CPP_NORMS = {"healthy_floor": 7.0, "severe_floor": 3.0}

# Expected syllable rate for "normal conversational speech"
EXPECTED_SYL_RATE = 4.5  # syllables per second


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float, midpoint: float, slope: float) -> float:
    z = np.clip(slope * (x - midpoint), -500, 500)
    return float(1.0 / (1.0 + np.exp(-z)))


def _tension_curve(raw_strain: float) -> float:
    """
    raw_strain 0-1 → tension 0-100.
    Normal ~0.05-0.15 → 3-15 tension.
    Growl  ~0.55-0.85 → 78-97 tension.
    """
    s = _sigmoid(raw_strain, midpoint=0.35, slope=10.0)
    return float(np.clip(s * 100, 0, 100))


def _vitality_from_tension(tension: float) -> float:
    """Asymmetric inverse: low tension → high vitality plateau."""
    raw = 1.0 - (tension / 100.0)
    return float(np.clip((raw ** 0.6) * 100, 0, 100))


def _cog_speed_curve(syllable_rate: float, completion_ratio: float) -> float:
    """
    Maps syllable rate to 0-100, penalised by completion ratio.
    If someone only spoke half the expected amount, cog_speed is halved.
    """
    if syllable_rate < 0.3:
        base = 8.0
    else:
        s = _sigmoid(syllable_rate, midpoint=2.5, slope=1.5)
        base = s * 88
        if 3.5 <= syllable_rate <= 5.5:
            base = max(base, 70 + (syllable_rate - 3.5) * 7.5)
        if syllable_rate > 6.0:
            base = min(base, 85 + 2 * np.log1p(syllable_rate - 6.0))

    # Completion penalty: if they only got through 40% of expected speech,
    # scale cog_speed down (but not below 30% of base — they still spoke)
    penalty = 0.3 + 0.7 * np.clip(completion_ratio, 0, 1)
    return float(np.clip(base * penalty, 0, 100))


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
# Cepstral Peak Prominence (CPP)
# ---------------------------------------------------------------------------

def _compute_cpp(y: np.ndarray, sr: int) -> float:
    """
    Cepstral Peak Prominence — the clinical gold standard for voice quality.

    Process:
    1. Compute the real cepstrum of the signal.
    2. Find the peak in the "pitch range" of the cepstrum (quefrency
       corresponding to 60-500 Hz).
    3. Fit a regression line to the cepstrum in that range.
    4. CPP = peak height above the regression line (in dB).

    High CPP (>8 dB)  = clear, periodic voice (normal).
    Low CPP  (<5 dB)  = aperiodic, noisy, or creaky (strained).
    """
    # Window the signal
    n_fft = 4096  # longer window for better quefrency resolution
    hop = 512

    # Process in frames and take the mean CPP
    cpps = []
    for start in range(0, len(y) - n_fft, hop):
        frame = y[start:start + n_fft]
        # Apply Hanning window
        frame = frame * np.hanning(n_fft)
        # Power spectrum
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        # Log power spectrum (add small epsilon)
        log_spec = np.log10(spectrum + 1e-20)
        # Cepstrum = IFFT of log power spectrum
        cepstrum = np.fft.irfft(log_spec)

        # Quefrency range for pitch: 60-500 Hz → quefrency 2ms-16.7ms
        q_min = int(sr / 500)   # ~44 samples at 22050
        q_max = int(sr / 60)    # ~367 samples at 22050
        q_max = min(q_max, len(cepstrum) - 1)

        if q_min >= q_max:
            continue

        ceps_region = cepstrum[q_min:q_max]
        quefrencies = np.arange(q_min, q_max)

        # Find peak
        peak_idx = np.argmax(ceps_region)
        peak_val = ceps_region[peak_idx]

        # Regression line through the cepstral region
        coeffs = np.polyfit(quefrencies, ceps_region, 1)
        regression_val = np.polyval(coeffs, quefrencies[peak_idx])

        # CPP = peak above regression, converted to dB-like scale
        cpp_frame = float(peak_val - regression_val)
        cpps.append(cpp_frame)

    if not cpps:
        return 0.0

    # Use the median (robust to outlier frames)
    return float(np.median(cpps))


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def _extract_features(y: np.ndarray, sr: int, gender: str) -> dict:
    """
    Five features designed to survive iPhone AGC:

    1. cpp              — Cepstral Peak Prominence (voice periodicity)
    2. f0_median        — Median fundamental frequency
    3. f0_irregularity  — Coefficient of variation + unvoiced ratio
    4. spectral_tilt    — Low-to-high energy ratio
    5. syllable_rate    — Articulatory events per second
    + completion_ratio  — Fraction of expected speech produced
    """
    features: dict = {}
    n_fft = 2048
    hop = 512

    # ── 1. CPP ──────────────────────────────────────────────────────────
    cpp = _compute_cpp(y, sr)
    features["cpp"] = round(cpp, 4)

    # ── 2 & 3. F0 Analysis via PYIN ────────────────────────────────────
    # PYIN is more robust than YIN for noisy/creaky voice
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr, hop_length=hop,
        fill_na=0.0  # unvoiced frames get 0
    )

    # Separate voiced frames
    voiced_f0 = f0[voiced_flag]
    n_total = len(f0)
    n_voiced = len(voiced_f0)

    if n_voiced > 3:
        f0_median = float(np.median(voiced_f0))
        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_cv = f0_std / (f0_mean + 1e-10)  # coefficient of variation
    else:
        f0_median = 0.0
        f0_mean = 0.0
        f0_cv = 1.0  # maximally irregular

    # Unvoiced ratio: fraction of frames where PYIN couldn't find a pitch
    # Normal speech: ~20-40% unvoiced (consonants, pauses).
    # Vocal fry: often 40-70% because the irregular pulses confuse PYIN.
    unvoiced_ratio = 1.0 - (n_voiced / (n_total + 1e-10))

    # Combined irregularity: CV + excess unvoiced penalty
    # Normal: CV ~0.10-0.25, unvoiced ~0.25-0.40
    # Growl:  CV ~0.30-0.60, unvoiced ~0.45-0.70
    f0_irregularity = 0.5 * np.clip(f0_cv / 0.40, 0, 1) + \
                      0.5 * np.clip((unvoiced_ratio - 0.30) / 0.35, 0, 1)

    features["f0_median"] = round(f0_median, 1)
    features["f0_mean"] = round(f0_mean, 1)
    features["f0_cv"] = round(f0_cv, 4)
    features["unvoiced_ratio"] = round(unvoiced_ratio, 4)
    features["f0_irregularity"] = round(float(f0_irregularity), 4)

    # ── 4. Spectral Tilt (Low-to-High Energy Ratio) ────────────────────
    # Even with AGC, the RATIO of energy in bands is preserved.
    # Growl: massive energy below 500 Hz, little above 2 kHz.
    # Normal: balanced spread.
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    low_mask = freqs <= 500
    high_mask = freqs >= 2000

    low_energy = float(np.mean(S[low_mask, :])) + 1e-20
    high_energy = float(np.mean(S[high_mask, :])) + 1e-20

    # Ratio in dB: positive = more low energy (growl-like)
    spectral_tilt_db = 10 * np.log10(low_energy / high_energy)
    features["spectral_tilt_db"] = round(spectral_tilt_db, 2)

    # Normal speech: tilt ~5-15 dB (some low dominance is natural)
    # Growl: tilt ~20-40+ dB (extreme low dominance)
    # Normalise to 0-1 strain: 12→0, 30→1
    tilt_strain = np.clip((spectral_tilt_db - 12) / 18, 0, 1)
    features["tilt_strain"] = round(float(tilt_strain), 4)

    # ── 5. F0 Depth Score ──────────────────────────────────────────────
    # How abnormally low is the F0 for this gender?
    norms = F0_NORMS.get(gender, F0_NORMS["other"])
    if f0_median > 0:
        # Score: how far below the "low threshold" is the F0?
        # At or above threshold → 0. At very_low → 1.
        threshold = norms["low_threshold"]
        very_low = norms["very_low"]
        if f0_median >= threshold:
            f0_depth = 0.0
        elif f0_median <= very_low:
            f0_depth = 1.0
        else:
            f0_depth = (threshold - f0_median) / (threshold - very_low)
    else:
        f0_depth = 0.8  # no pitch detected → likely very abnormal

    features["f0_depth"] = round(float(f0_depth), 4)

    # ── 6. Syllable Rate + Completion ──────────────────────────────────
    mel_S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=80
    )
    mel_db = librosa.power_to_db(mel_S, ref=np.max)
    flux = np.sum(np.maximum(0, np.diff(mel_db, axis=1)), axis=0)

    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-10
    threshold = med + 1.5 * mad

    min_gap_frames = max(1, int(0.12 * sr / hop))
    peaks = []
    last = -min_gap_frames - 1
    for i in range(1, len(flux) - 1):
        if (flux[i] > threshold
                and flux[i] >= flux[i - 1]
                and flux[i] >= flux[i + 1]
                and (i - last) >= min_gap_frames):
            peaks.append(i)
            last = i

    duration_sec = len(y) / sr
    n_onsets = len(peaks)
    syllable_rate = n_onsets / duration_sec if duration_sec > 0.1 else 0.0

    # Completion ratio: how many syllables were produced vs. expected
    # for this duration? If someone records 5 seconds, we'd expect
    # ~22 syllables (4.5/s * 5s). If only 8 were detected, ratio ≈ 0.36.
    expected_syllables = EXPECTED_SYL_RATE * duration_sec
    completion_ratio = n_onsets / (expected_syllables + 1e-10)

    features["n_onsets"] = n_onsets
    features["duration_sec"] = round(duration_sec, 2)
    features["syllable_rate"] = round(syllable_rate, 2)
    features["completion_ratio"] = round(float(np.clip(completion_ratio, 0, 1.5)), 4)

    return features


# ---------------------------------------------------------------------------
# Composite Scoring
# ---------------------------------------------------------------------------

def _compute_scores(features: dict, is_smoker: bool, gender: str) -> dict:
    """
    Tension composite from 4 strain indicators:

    1. CPP deficit     (weight 0.30) — THE clinical voice quality measure
    2. F0 depth        (weight 0.25) — abnormally low pitch
    3. F0 irregularity (weight 0.25) — erratic pitch + voicing dropouts
    4. Spectral tilt   (weight 0.20) — low-frequency energy dominance

    Why this works when v1/v2 didn't:
    - CPP is a RATIO measure — AGC can't hide aperiodicity
    - F0 is tracked by autocorrelation — AGC shifts level, not frequency
    - Spectral tilt is an energy RATIO — AGC scales uniformly
    - F0 irregularity catches the erratic pitch jumps of vocal fry
    """
    cpp = features["cpp"]
    f0_depth = features["f0_depth"]
    f0_irreg = features["f0_irregularity"]
    tilt_strain = features["tilt_strain"]
    syl_rate = features["syllable_rate"]
    completion = features["completion_ratio"]

    # ── CPP deficit (0-1) ──
    # CPP healthy_floor=7.0, severe_floor=3.0
    # Above 7 → deficit 0. Below 3 → deficit 1.
    if cpp >= CPP_NORMS["healthy_floor"]:
        cpp_deficit = 0.0
    elif cpp <= CPP_NORMS["severe_floor"]:
        cpp_deficit = 1.0
    else:
        cpp_deficit = (CPP_NORMS["healthy_floor"] - cpp) / \
                      (CPP_NORMS["healthy_floor"] - CPP_NORMS["severe_floor"])

    # ── Composite raw strain ──
    raw_strain = (
        0.30 * float(cpp_deficit)
        + 0.25 * float(f0_depth)
        + 0.25 * float(f0_irreg)
        + 0.20 * float(tilt_strain)
    )

    if is_smoker:
        raw_strain += 0.05

    tension = _tension_curve(raw_strain)
    vitality = _vitality_from_tension(tension)
    cog_speed = _cog_speed_curve(syl_rate, completion)

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
            "cpp_deficit": round(float(cpp_deficit), 4),
            "f0_depth": round(float(f0_depth), 4),
            "f0_irregularity": round(float(f0_irreg), 4),
            "tilt_strain": round(float(tilt_strain), 4),
            "is_smoker": is_smoker,
            "gender": gender,
        },
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=BiomarkerResult)
async def analyze_voice(
    file: UploadFile = File(..., description="Audio file"),
    is_smoker: bool = Form(False),
    gender: str = Form("other"),
):
    """
    Upload audio via multipart/form-data.

    Frontend example:
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        formData.append('is_smoker', isSmoker ? 'true' : 'false');
        formData.append('gender', gender || 'other');
        const res = await fetch('/analyze', { method: 'POST', body: formData });
        // Do NOT set Content-Type header — browser handles boundary
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
    log.info("Features: %s", features)

    scores = _compute_scores(features, is_smoker, gender)
    log.info(
        "vrs=%.1f tension=%.1f vitality=%.1f cog=%.1f",
        scores["vrs"], scores["tension"], scores["vitality"], scores["cog_speed"],
    )

    return BiomarkerResult(**scores)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
