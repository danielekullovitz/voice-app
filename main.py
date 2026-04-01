import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
async def analyze_voice(
    file: UploadFile = File(...),
    is_smoker: str = Form("false"),
    gender: str = Form("unspecified")
):
    unique_id = str(uuid.uuid4())
    temp_path = f"temp_{unique_id}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. THE VOICE PURIFIER
        y, sr = librosa.load(temp_path, sr=None)
        y_filt = librosa.effects.preemphasis(y)
        y_voice, _ = librosa.effects.trim(y_filt, top_db=20) 

        if len(y_voice) < sr * 0.5: 
            y_voice = y_filt

        # --- ENGINE V16: THE ARTICULATION ENGINE ---

        # 1. TENSION (Throat/Noise)
        # Median flatness ignores harsh consonants and focuses on the core vocal tone.
        flatness_array = librosa.feature.spectral_flatness(y=y_voice)[0]
        median_flatness = np.median(flatness_array)
        
        smoker_flag = is_smoker.lower() in ['true', '1', 'yes']
        base_hiss_tax = 0.0100 if smoker_flag else 0.0050
        clean_flatness = max(0, median_flatness - base_hiss_tax)
        
        # Massive multiplier because normal median flatness is tiny (~0.001)
        tension = int(min(100, max(5, clean_flatness * 6000)))

        # 2. VITALITY (Energy & Expression)
        # Normal speech has high dynamic range (pauses, loud/soft). Monotone strain has low.
        rms_array = librosa.feature.rms(y=y_voice)[0]
        dynamic_range = np.std(rms_array) / np.mean(rms_array) if np.mean(rms_array) > 0 else 0
        
        vitality_calc = 100 - tension + (dynamic_range * 15)
        vitality = int(min(100, max(10, vitality_calc)))

        # 3. COG SPEED (Brain/Articulation)
        # Count actual word attacks (syllables) per second. Ignores tone entirely!
        onsets = librosa.onset.onset_detect(y=y_voice, sr=sr, backtrack=False)
        duration = len(y_voice) / sr
        articulation_rate = len(onsets) / duration if duration > 0 else 0
        
        # Normal conversational speech is ~3.5 to 5 onsets per second.
        cog_speed_calc = int(articulation_rate * 22)
        cog_speed = int(min(100, max(10, cog_speed_calc)))

        # 4. FINAL VRS
        vrs_score = int(((100 - tension) * 0.6) + (vitality * 0.2) + (cog_speed * 0.2))
        vrs_score = max(5, min(98, vrs_score))

        result = {
            "vrs": vrs_score,
            "tension": tension,
            "vitality": vitality,
            "cog_speed": cog_speed,
            "meta": {
                "median_flatness": f"{median_flatness:.4f}",
                "dynamic_range": f"{dynamic_range:.2f}",
                "articulation_rate": f"{articulation_rate:.2f}"
            }
        }
        
        print(f"--- ENGINE V16 COMPLETE | VRS: {vrs_score} ---")
        return result

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
