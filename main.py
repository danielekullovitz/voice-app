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
            https://id-preview--9c404639-9fe2-4b60-863c-466d1a5ad956.lovable.app
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
