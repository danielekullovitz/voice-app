import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_voice(
    file: UploadFile = File(...),
    is_smoker: str = Form("false"),
    gender: str = Form("unspecified")
):
    unique_id = str(uuid.uuid4())
    temp_path = f"temp_{unique_id}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. THE VOICE PURIFIER
        y, sr = librosa.load(temp_path, sr=None)
        y_filt = librosa.effects.preemphasis(y)
        y_voice, _ = librosa.effects.trim(y_filt, top_db=20) 

        if len(y_voice) < sr * 0.5: 
            y_voice = y_filt

        # 2. BIO-METRIC EXTRACTION
        f0, _, _ = librosa.pyin(y_voice, fmin=75, fmax=600)
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else []
        avg_f0 = np.mean(f0_clean) if len(f0_clean) > 0 else 0

        flatness = np.mean(librosa.feature.spectral_flatness(y=y_voice))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_voice))
        rms_array = librosa.feature.rms(y=y_voice)[0]
        shimmer = np.std(rms_array) / np.mean(rms_array) if np.mean(rms_array) > 0 else 0
        
        # NEW: Cognitive Extraction (Zero-Crossing Rate)
        # Measures how fast the vocal tract is changing shapes (articulation)
        zcr_array = librosa.feature.zero_crossing_rate(y=y_voice)[0]
        zcr_var = np.var(zcr_array) * 1000  # Scale it up so we can use it

        # --- 3. ENGINE V13: DECOUPLED METRICS ---
        smoker_flag = is_smoker.lower() in ['true', '1', 'yes']
        base_hiss_tax = 0.0250 if smoker_flag else 0.0150
        clean_flatness = max(0, flatness - base_hiss_tax)
        
        # TENSION (The Throat): Purely physical strain and noise.
        tension_raw = (shimmer * 160) + (clean_flatness * 2000) + (bandwidth / 120)
        tension = int(min(100, max(10, tension_raw - 55)))
        
        # VITALITY (The Lungs/Vocal Folds): Inverse of physical strain.
        vitality_calc = 130 - (shimmer * 350) - (tension * 0.6)
        vitality = int(min(100, max(15, vitality_calc)))
        
        # COG SPEED (The Brain): Based on dynamic articulation (ZCR Variance)
        # If you are speaking words, ZCR variance is high. If you are doing a flat growl, it drops.
        # We completely untie this from Tension!
        cog_speed_calc = 40 + (zcr_var * 15) 
        cog_speed = int(min(100, max(20, cog_speed_calc)))

        # VRS: The True North Score
        vrs_score = int(((100 - (tension * 1.1)) + vitality) / 2)
        vrs_score = max(5, min(98, vrs_score))

        result = {
            "vrs": vrs_score,
            "tension": tension,
            "vitality": vitality,
            "cog_speed": cog_speed,
            "meta": {
                "smoker_adjusted": smoker_flag,
                "gender": gender,
                "zcr_variance": f"{zcr_var:.4f}"
            }
        }
        
        print(f"--- ENGINE V13 COMPLETE | VRS: {vrs_score} ---")
        return result

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
