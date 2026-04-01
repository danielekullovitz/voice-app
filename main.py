import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
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
async def analyze_voice(file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    temp_path = f"temp_{unique_id}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. THE VOICE PURIFIER (iPhone Optimization)
        y, sr = librosa.load(temp_path, sr=None)
        
        # Pre-emphasis (Sharps the vocal signal, dulls the hiss)
        y_filt = librosa.effects.preemphasis(y)
        
        # Strict Trim (Ignore silence or handling noise)
        y_voice, _ = librosa.effects.trim(y_filt, top_db=20) 

        # CRITICAL FIX: If the trim cuts too much (e.g., quiet recording), fallback to original
        if len(y_voice) < sr * 0.5: 
            y_voice = y_filt

        # 2. BIO-METRIC EXTRACTION
        # Silently extract Pitch (F0) for future database profiling
        f0, _, _ = librosa.pyin(y_voice, fmin=75, fmax=600)
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else []
        avg_f0 = np.mean(f0_clean) if len(f0_clean) > 0 else 0

        # Purity Metrics
        flatness = np.mean(librosa.feature.spectral_flatness(y=y_voice))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_voice))
        
        # Stability Metrics
        rms_array = librosa.feature.rms(y=y_voice)[0]
        shimmer = np.std(rms_array) / np.mean(rms_array) if np.mean(rms_array) > 0 else 0

        # 3. ENGINE V10 LOGIC: BULLETPROOF CALIBRATION
        clean_flatness = max(0, flatness - 0.0150)
        
        # TENSION
        tension_raw = (shimmer * 120) + (clean_flatness * 2500) + (bandwidth / 150)
        tension = int(min(100, max(12, tension_raw - 45)))
        
        # VITALITY
        vitality_calc = 135 - (clean_flatness * 3000) - (bandwidth / 60)
        vitality = int(min(100, max(15, vitality_calc)))
        
        # VRS: The Health Score
        vrs_score = int(((100 - (tension * 1.1)) + vitality) / 2)
        vrs_score = max(5, min(98, vrs_score))
        
        # COG SPEED
        cog_speed = int(max(40, 102 - (clean_flatness * 2800)))

        result = {
            "vrs": vrs_score,
            "tension": tension,
            "vitality": vitality,
            "cog_speed": min(100, cog_speed),
            "meta": {
                "raw_flatness": f"{flatness:.4f}", 
                "pitch_hz": f"{avg_f0:.2f}",
                "mic_type": "iPhone Optimized v10"
            }
        }
        
        print(f"--- ENGINE V10 PRO COMPLETE: {result} ---")
        return result

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
