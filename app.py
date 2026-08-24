import os
import tempfile
import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Smart Cradle Audio Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained ML model
model = joblib.load("smart_cradle_model.pkl")


# Serve the web dashboard directly from root URL
@app.get("/")
async def serve_dashboard():
    return FileResponse("index.html")


def extract_features(file_path):
    # Load audio and extract 40 MFCC features
    audio, sr = librosa.load(file_path, sr=22050, duration=5.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.reshape(1, -1)


@app.post("/predict")
async def predict_cry(file: UploadFile = File(...)):
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[-1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name

        features = extract_features(temp_path)

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities)) * 100

        return {
            "status": "success",
            "detected_cry": prediction,
            "confidence": round(confidence, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)