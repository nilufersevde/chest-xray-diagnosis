import io
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# These transforms MUST match your training transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image_bytes, model, device="cpu", threshold=0.7):
    # Load image from raw bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    
    # Apply transforms
    image = transform(image).unsqueeze(0).to(device)  # shape: [1, 1, 224, 224]

    # Inference
    with torch.no_grad():
        outputs = model(image)
        print("Model raw output logits:", outputs)
        
        # Get probabilities
        probabilities = F.softmax(outputs, dim=1)
        print("Probabilities:", probabilities)
        
        # Get prediction with threshold
        normal_prob = probabilities[0][0].item()
        pneumonia_prob = probabilities[0][1].item()
        
        # Apply threshold-based decision to reduce bias
        if pneumonia_prob > threshold:
            predicted_class = 1  # PNEUMONIA
            confidence = pneumonia_prob
        elif normal_prob > threshold:
            predicted_class = 0  # NORMAL
            confidence = normal_prob
        else:
            # If neither class has high confidence, use the more balanced approach
            # Add a small bias towards normal to counteract the model's pneumonia bias
            adjusted_normal = normal_prob * 1.2  # Boost normal probability
            predicted_class = 0 if adjusted_normal > pneumonia_prob else 1
            confidence = max(normal_prob, pneumonia_prob)
        
        print(f"Predicted class index: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Threshold applied: {threshold}")

    # Convert index to label
    class_names = ["NORMAL", "PNEUMONIA"]
    prediction = class_names[predicted_class]
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "NORMAL": normal_prob,
            "PNEUMONIA": pneumonia_prob
        },
        "threshold_used": threshold
    } 
