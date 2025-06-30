import torch
import torch.nn.functional as F
from model.load_model import load_trained_model
from utils.predict import predict_image
from PIL import Image
import io

def test_model_with_sample():
    """Test the model with a simple test image to debug predictions"""
    
    # Load the model
    model = load_trained_model()
    model.eval()
    
    # Create a simple test image (all zeros - should be easy to classify)
    test_image = Image.new('L', (224, 224), color=128)  # Gray image
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Test prediction
    print("Testing with a simple gray image...")
    result = predict_image(img_byte_arr, model)
    
    print("\n=== PREDICTION RESULTS ===")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Normal probability: {result['probabilities']['NORMAL']:.4f}")
    print(f"Pneumonia probability: {result['probabilities']['PNEUMONIA']:.4f}")
    
    # Test with a white image
    test_image_white = Image.new('L', (224, 224), color=255)
    img_byte_arr_white = io.BytesIO()
    test_image_white.save(img_byte_arr_white, format='PNG')
    img_byte_arr_white = img_byte_arr_white.getvalue()
    
    print("\nTesting with a white image...")
    result_white = predict_image(img_byte_arr_white, model)
    
    print("\n=== PREDICTION RESULTS (WHITE) ===")
    print(f"Prediction: {result_white['prediction']}")
    print(f"Confidence: {result_white['confidence']:.4f}")
    print(f"Normal probability: {result_white['probabilities']['NORMAL']:.4f}")
    print(f"Pneumonia probability: {result_white['probabilities']['PNEUMONIA']:.4f}")

if __name__ == "__main__":
    test_model_with_sample() 