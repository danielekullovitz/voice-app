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
               # --- ENGINE V4: THE BIOMETRIC GOLD STANDARD ---
        
        # 1. Jitter & Shimmer (Standardized)
        jitter_score = min(100, (jitter * 60))
        shimmer_score = min(100, (shimmer * 50))
        
        # 2. Tension: Now heavily weighted by 'Flatness' (the Growl detector)
        # Flatness represents noise. Growls have high flatness.
        tension_raw = (jitter_score * 0.3) + (shimmer_score * 0.2) + (flatness * 600)
        tension = int(min(100, max(10, tension_raw)))
        
        # 3. Vitality: Rewards a clear, stable voice. 
        # We subtract Tension from Vitality so you CANNOT have a high score if you are straining.
        base_vitality = 100 - (flatness * 500) - (shimmer_score * 0.5)
        vitality = int(base_vitality - (tension * 0.2)) # Tension "drains" vitality
        vitality = max(20, min(100, vitality))
        
        # 4. FINAL VRS: The true "Health" score
        vrs_score = int((100 - tension) * 0.5 + vitality * 0.5)

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
