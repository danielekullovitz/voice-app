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
        # Get Fundamental Frequency (Pitch)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=600)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            avg_f0 = 0
            jitter = 0
        else:
            avg_f0 = np.mean(f0_clean)
            # Calculate Jitter (Stability of pitch)
            jitter = np.std(f0_clean) / avg_f0 if avg_f0 > 0 else 0

        # Calculate Shimmer (Stability of amplitude/loudness)
        rms = librosa.feature.rms(y=y)[0]
        shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0

        # Calculate Spectral Flatness (Crucial for detecting growls/rasp)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        # --- 2. ENGINE V6: SPECTRUM DOMINANCE ---
        
        # TENSION: Low Jitter sensitivity (human speech) + HUGE Flatness penalty (noise/growls)
        tension_calc = (jitter * 150) + (flatness * 2000) 
        tension = int(min(100, max(15, tension_calc)))
        
        # VITALITY: Bonus for "Clear" tone, penalty for "Noise"
        vitality_calc = 110 - (shimmer * 300) - (flatness * 1500)
        vitality = int(min(100, max(10, vitality_calc)))
        
        # THE "VOICE HEALTH" WEIGHTING (VRS)
        # Tension is the score killer here.
        vrs_score = int(((100 - (tension * 1.2)) + vitality) / 2)
        vrs_score = max(5, min(98, vrs_score))
        
        # COG SPEED (Based on mental processing reflected in pitch stability)
        cog_speed = int(max(40, 100 - (jitter * 300)))

        result = {
            "vrs": vrs_score,
            "tension": tension,
            "vitality": vitality,
            "cog_speed": cog_speed,
            "jitter": f"{jitter:.2%}",
            "shimmer": f"{shimmer:.2%}",
            "flatness": f"{flatness:.4f}",
            "f0": f"{avg_f0:.2f} Hz"
        }
        
        print(f"--- REAL-TIME ANALYSIS COMPLETE: {result} ---")
        return result

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the file so Railway doesn't run out of storage
        if os.path.exists(temp_path):
            os.remove(temp_path)
