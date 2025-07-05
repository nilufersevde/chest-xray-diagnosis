from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from backend.model.load_model import load_trained_model
from backend.utils.predict import predict_image
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Mount static files (CSS, JS) from the frontend directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load model when the app starts
model = load_trained_model()
device = "cpu"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("frontend/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file bytes
    image_bytes = await file.read()
    # Get prediction
    result = predict_image(image_bytes, model, device)
    # Return JSON response
    return JSONResponse(content=result)

@app.get("/health")
def health_check():
    return {"status": "API is running"}
