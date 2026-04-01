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
    # Create a UNIQUE filename for every single scan
    unique_id = str(uuid.uuid4())
    temp_path = f"temp_{unique_id}_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load audio
        y, sr = librosa.load(temp_path, sr=None)
        
        # --- 1. CORE ACOUSTIC EXTRACTION ---
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=600)
        f0_clean = f0[~np.isnan(f0)]
        avg_f0 = np.mean(f0_clean) if len(f0_clean) > 0 else 0
        
        # Stability & Purity Measures
        rms = librosa.feature.rms(y=y)[0]
        shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y))

        # --- 2. ENGINE V7: THE HARMONIC HERO ---
        
        # TENSION: We focus on Shimmer (crackling) and Bandwidth (noise/growl).
        # A growl is "wide" noise (High Bandwidth). A clear voice is "narrow" (Low Bandwidth).
        tension_calc = (shimmer * 200) + (flatness * 1800) + (bandwidth / 80)
        # Calibration shift: Normal voice should sit at 15-25 Tension.
        tension = int(min(100, max(15, tension_calc - 35)))
        
        # VITALITY: A clear voice is "Peaky". 
        # High Flatness (noise) and High Bandwidth (messiness) CRUSH vitality.
        vitality_calc = 125 - (flatness * 2200) - (bandwidth / 40)
        vitality = int(min(100, max(10, vitality_calc)))
        
        # VRS: The Final Score
        # We weight Tension heavily. If you have 80 Tension, your score is dead.
        vrs_score = int(((100 - tension) * 0.75) + (vitality * 0.25))
        vrs_score = max(5, min(98, vrs_score))
        
        # COG SPEED: Based on Spectral Flatness (Clarity).
        # "Clear Voice = Clear Mind."
        cog_speed = int(max(30, 98 - (flatness * 2500)))

        result = {
            "vrs": vrs_score,
            "tension": tension,
            "vitality": vitality,
            "cog_speed": cog_speed,
            "debug": {
                "flatness": f"{flatness:.4f}",
                "bandwidth": f"{bandwidth:.2f}",
                "shimmer": f"{shimmer:.2%}"
            }
        }
        
        print(f"--- ENGINE V7 COMPLETE: {result} ---")
        return result

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the file
        if os.path.exists(temp_path):
            os.remove(temp_path)

