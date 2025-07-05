# Chest X-ray Pneumonia Classifier

A deep learning-powered web application for automated chest X-ray analysis and pneumonia detection. Built with FastAPI, PyTorch, and vanilla HTML/CSS/JavaScript, this application provides real-time medical image analysis with confidence scoring.

## Overview

This application uses a Convolutional Neural Network (CNN) trained on a balanced dataset of **pediatric chest X-ray images** to classify between normal and pneumonia cases. The model achieves 85% accuracy on the test set and provides confidence scores for predictions.

### Key Features

- **Pediatric X-ray Analysis**: Optimized for analyzing chest X-rays from children and adolescents
- **Real-time X-ray Analysis**: Upload chest X-ray images and get instant predictions
- **Confidence Scoring**: Each prediction includes confidence levels and probability distributions
- **Uncertainty Handling**: Returns "UNCERTAIN" for low-confidence predictions
- **Professional UI**: Clean, responsive web interface
- **Production Ready**: Deployed on Railway with full-stack integration
- **Medical Domain Expertise**: Built by an MD with CSE background

## Live Demo

[View Live Application](https://chest-xray-backend-production.up.railway.app/)

## Technology Stack

### Backend
- **FastAPI**: High-performance web framework for building APIs
- **PyTorch**: Deep learning framework for model inference
- **PIL**: Image processing and preprocessing
- **NumPy**: Numerical computations

### Frontend
- **HTML/CSS/JavaScript**: Clean, responsive web interface
- **Static File Serving**: Integrated frontend-backend deployment
- **Vanilla JavaScript**: No framework dependencies

### Machine Learning
- **CNN Architecture**: Custom SimpleCNN with 2 convolutional layers
- **Transfer Learning**: Pre-trained on medical imaging datasets
- **Data Augmentation**: Balanced training with class weights
- **Model Performance**: 85% accuracy, 84% F1-score

### Deployment
- **Railway**: Cloud deployment platform
- **GitHub**: Version control and CI/CD
- **Docker**: Containerized deployment (if needed)

## Model Performance

```
Confusion Matrix:
[[184  50]
 [ 43 347]]

Classification Report:
              precision    recall  f1-score   support
      NORMAL       0.81      0.79      0.80       234
   PNEUMONIA       0.87      0.89      0.88       390

    accuracy                           0.85       624
   macro avg       0.84      0.84      0.84       624
weighted avg       0.85      0.85      0.85       624
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   PyTorch       │
│   (HTML/CSS/JS) │◄──►│   Backend       │◄──►│   CNN Model     │
│                 │    │                 │    │                 │
│ • Image Upload  │    │ • File Handling │    │ • Inference     │
│ • Results Display│   │ • Preprocessing │    │ • Confidence    │
│ • Responsive UI │    │ • API Endpoints │    │ • Predictions   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

##  Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- FastAPI
- PIL (Pillow)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/chest-xray-diagnosis.git
   cd chest-xray-diagnosis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the application**
   - Open your browser and go to `http://localhost:8000`
   - Upload a chest X-ray image
   - View the prediction results

## Project Structure

```
chest-xray-diagnosis/
├── backend/
│   ├── main.py                 # FastAPI application
│   │   ├── load_model.py       # Model loading utilities
│   │   └── model_balanced.pth  # Trained model weights
│   ├── utils/
│   │   └── predict.py          # Prediction utilities
│   └── scripts/
│       ├── evaluate_model.py   # Model evaluation
│       └── retrain_balanced.py # Model training
├── frontend/
│   ├── index.html              # Main web interface
│   ├── styles.css              # Styling
│   └── script.js               # Frontend logic
├── notebooks/
│   └── 01_eda_and_preprocessing.ipynb  # Data analysis
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

##  API Endpoints

### POST `/predict`
Upload a chest X-ray image for analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "prediction": "NORMAL",
  "confidence": 0.92,
  "probabilities": {
    "NORMAL": 0.92,
    "PNEUMONIA": 0.08
  },
  "threshold_used": 0.85
}
```

### GET `/`
Serves the main web interface.

### GET `/health`
Health check endpoint.

## Model Details

### Architecture
- **Input**: 224x224 grayscale pediatric chest X-ray images
- **Preprocessing**: Normalization, resizing, grayscale conversion
- **Architecture**: SimpleCNN with 2 convolutional layers
- **Output**: Binary classification (Normal vs Pneumonia)
- **Target Population**: Children and adolescents (pediatric cases)

### Training
- **Dataset**: Pediatric Chest X-ray Dataset (Kermany et al.) from Guangzhou Women and Children's Medical Center, available via Kaggle
- **Balanced Training**: Class weights to handle imbalance
- **Data Augmentation**: Random flips, rotations
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Cross-entropy with class weights

## Key Features Explained

### Pediatric Focus
This application is specifically designed for pediatric chest X-ray analysis. The model was trained on chest X-rays from children and adolescents, making it particularly effective for pediatric cases. While it can analyze adult chest X-rays, the highest accuracy and reliability are achieved with pediatric images.

### Confidence Threshold
The application uses a confidence threshold of 0.85 to ensure reliable predictions. When the model's confidence is below this threshold, it returns "UNCERTAIN" rather than making potentially incorrect predictions.

### Uncertainty Handling
```python
if confidence < threshold:
    predicted_class = "UNCERTAIN"
```

### Real-time Processing
Images are processed in real-time with the following pipeline:
1. Image upload and validation
2. Preprocessing (resize, normalize, convert to grayscale)
3. Model inference
4. Confidence calculation
5. Result formatting and display

## Security & Privacy

- **No Data Storage**: Images are processed in memory and not stored
- **Input Validation**: File type and size validation
- **Error Handling**: Graceful error handling for invalid inputs
- **CORS Enabled**: Configured for web application access

## Deployment

The application is deployed on Railway with the following configuration:

- **Backend**: FastAPI application with static file serving
- **Frontend**: Integrated HTML/CSS/JavaScript
- **Model**: Pre-trained PyTorch model loaded at startup
- **Database**: No database required (stateless application)

## Future Enhancements

- [ ] Multi-disease classification (COVID-19, tuberculosis)
- [ ] Medical report generation
- [ ] User authentication and history
- [ ] Mobile application
- [ ] Integration with PACS systems
- [ ] Advanced visualization tools
