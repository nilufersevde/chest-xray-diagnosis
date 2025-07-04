from pathlib import Path
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available. Model loading will be skipped.")
    TORCH_AVAILABLE = False

# Define the same model architecture you trained
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        image_size = 224
        self.fc1 = nn.Linear(32 * (image_size // 4) * (image_size // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_trained_model(device="cpu"):
    """
    Load the trained model. Now uses the balanced model by default.
    You can specify a different path if needed.
    """
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Cannot load model.")
        return None

    # Figure out the directory this file is in (backend/model)
    this_dir = Path(__file__).resolve().parent

    # Build absolute paths to both models
    balanced_model_path = this_dir / "model_balanced.pth"
    fallback_model_path = this_dir / "model.pth"

    model = SimpleCNN()

    try:
        model.load_state_dict(torch.load(balanced_model_path, map_location=device))
        print(f"✅ Loaded balanced model from {balanced_model_path}")
    except FileNotFoundError:
        print(f"⚠️ Balanced model not found at {balanced_model_path}, trying original model...")
        try:
            model.load_state_dict(torch.load(fallback_model_path, map_location=device))
            print(f"✅ Loaded original model from {fallback_model_path}")
        except FileNotFoundError:
            print(f"❌ No model found at {balanced_model_path} or {fallback_model_path}")
            raise

    model.to(device)
    model.eval()
    return model