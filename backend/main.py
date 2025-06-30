from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from model.load_model import load_trained_model
from utils.predict import predict_image


# Initialize FastAPI app
app = FastAPI()

# Load model when the app starts
model = load_trained_model()
device = "cpu"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file bytes
    image_bytes = await file.read()
    # Get prediction
    result = predict_image(image_bytes, model, device)
    # Return JSON response
    return JSONResponse(content=result)

@app.get("/")
def read_root():
    return {"status": "API is running"}
