import io
import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smart Cradle Audio Processor")

# Allow your web dashboard to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained ML model
model = joblib.load("smart_cradle_model.pkl")


def extract_features_from_audio(file_bytes):
    # Load audio directly from memory buffer
    audio, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, duration=5.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.reshape(1, -1)


@app.post("/predict")
async def predict_cry(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # Extract MFCCs
    features = extract_features_from_audio(audio_bytes)

    # Run ML inference
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities)) * 100

    return {
        "status": "success",
        "detected_cry": prediction,
        "confidence": round(confidence, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)