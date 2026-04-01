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
    temp_path = f"temp_{file.filename}"
    
    try:
        # 1. Save uploaded file locally (more stable)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Load audio with librosa
        y, sr = librosa.load(temp_path, sr=None)
        
        # 3. Biomarker Extraction
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        f0 = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 120.0
        
        # Jitter & Shimmer Logic
        jitter = np.std(librosa.feature.mfcc(y=y, sr=sr)[0]) / 100 
        shimmer = np.std(librosa.feature.rms(y=y)[0]) / 10
        
        # Calculate Scores
               # --- TUNED CALIBRATION ---
        # We lower the multipliers (from 500/200 to 150/100) so the score is less 'jumpy'
               # --- ENGINE V3: BIOLOGICAL BALANCE ---
        # 1. More forgiving Jitter/Shimmer (lower multipliers)
        jitter_score = min(100, (jitter * 80)) # Was 150
        shimmer_score = min(100, (shimmer * 60)) # Was 100
        
        # 2. Tension is the average of instabilities
        tension = int((jitter_score + shimmer_score) / 2)
        
        # 3. RELATIONAL VITALITY: 
        # High tension now "stifles" vitality. You can't have 99 Vitality if Tension is 100.
        raw_vitality = 100 - (shimmer * 120)
        vitality = int(raw_vitality * (1 - (tension / 250))) # Tension subtracts up to 40% of vitality
        vitality = max(40, min(100, vitality)) # Keep it in a realistic range
        
        # 4. FINAL VRS (The Balance)
        vrs_score = int((100 - tension) * 0.4 + vitality * 0.6)

        # New VRS Weighting: Vitality carries more weight for a "Wellness" feel
        vrs_score = int((100 - tension) * 0.3 + vitality * 0.7)

        return {
            "vrs": vrs_score,
            "tension": int(tension),
            "vitality": int(vitality),
            "jitter": f"{jitter:.2%}",
            "shimmer": f"{shimmer:.2%}",
            "f0": f"{f0:.2f} Hz"
        }

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up the file so the server doesn't get cluttered
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def health_check():
    return {"status": "Voxis Engine Online"}
