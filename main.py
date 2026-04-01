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
import uuid # Add this to the very top of your file with other imports

@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    # Create a UNIQUE filename for every single scan
    unique_id = str(uuid.uuid4())
    temp_path = f"temp_{unique_id}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        y, sr = librosa.load(temp_path, sr=None)
        
        # --- ADVANCED BIOMARKERS ---
        # 1. Jitter (Pitch Instability)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        f0 = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 120.0
        jitter = np.std(librosa.feature.mfcc(y=y, sr=sr)[0]) / 100
        
        # 2. Shimmer (Amplitude Instability)
        shimmer = np.std(librosa.feature.rms(y=y)[0]) / 10
        
        # 3. HARSHNESS (The "Monster Growl" Detector)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # --- TUNED CALIBRATION ---
        # If flatness is high (growly/noisy), Tension spikes automatically
        tension_raw = (jitter * 100) + (shimmer * 100) + (flatness * 500)
        tension = int(min(100, max(15, tension_raw)))
        
        # Vitality is "Brightness" - Growls are dark, so Vitality should drop
        vitality = int(100 - (flatness * 400) - (shimmer * 100))
        vitality = max(30, min(100, vitality))
        
        vrs_score = int((100 - tension) * 0.4 + vitality * 0.6)

        result = {
            "vrs": vrs_score,
            "tension": tension,
            "vitality": vitality,
            "jitter": f"{jitter:.2%}",
            "shimmer": f"{shimmer:.2%}",
            "f0": f"{f0:.2f} Hz"
        }
        
        print(f"--- REAL-TIME ANALYSIS COMPLETE: {result} ---")
        return result

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
