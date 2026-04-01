from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np

app = FastAPI()

# Allow Lovable to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    # 1. Load the audio
    y, sr = librosa.load(file.file, sr=None)
    
    # 2. Extract Real Biomarkers (Simplified for v1)
    # Pitch (Fundamental Frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    f0 = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 120.0
    
    # Jitter (Short-term pitch instability)
    jitter = np.std(librosa.feature.mfcc(y=y, sr=sr)[0]) / 100 
    
    # Shimmer (Amplitude instability)
    shimmer = np.std(librosa.feature.rms(y=y)[0]) / 10
    
    # 3. Calculate the VRS (The "Voxis" Secret Sauce)
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
