
from fastapi import FastAPI, UploadFile, File
from app.predict import load_model, run_inference

app = FastAPI()
model = load_model()

@app.get("/")
def root():
    return {"message": "YOLOv8 food classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = await run_inference(file, model)
    return result
