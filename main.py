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
        tension = min(100, max(0, (jitter * 500) + (shimmer * 200)))
        vitality = 100 - (shimmer * 500)
        vrs_score = int((100 - tension) * 0.4 + vitality * 0.6)

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
