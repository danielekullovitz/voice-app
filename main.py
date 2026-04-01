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
        
                # --- ENGINE V5: THE SCORE KILLER ---
        
        # 1. TENSION (The Penalty)
        # We multiply Flatness by 1000 to make sure growls spike Tension instantly.
        tension_calc = (jitter * 400) + (flatness * 1000)
        tension = int(min(100, max(15, tension_calc)))
        
        # 2. VITALITY (The Power)
        # A growl is "noisy," so high flatness must CRUSH vitality.
        vitality_calc = 100 - (shimmer * 250) - (flatness * 800)
        vitality = int(min(100, max(10, vitality_calc)))
        
        # 3. VRS (The Final Health Score)
        # Formally: (Healthy Buffer + Vitality) / 2
        # If Tension is 80, your "Healthy Buffer" is only 20.
        vrs_score = int(((100 - tension) + vitality) / 2)
        
        # 4. COG SPEED (Just for the UI - based on pitch stability)
        cog_speed = int(max(40, 100 - (jitter * 500)))

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
