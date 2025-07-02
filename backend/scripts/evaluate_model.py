print("✅ Script started")
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# 1) Re-create the model architecture
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        image_size = 224
        self.fc1 = torch.nn.Linear(32 * (image_size // 4) * (image_size // 4), 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2) Load the trained model weights
model = SimpleCNN()
model.load_state_dict(torch.load("backend/model/model_balanced.pth"))
model.eval()

# 3) Define the transforms (must match training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 4) Load the test dataset
dataset_path = Path("/Users/nilufersevdeozdemir/Downloads/chest_xray")
test_dataset = datasets.ImageFolder(dataset_path / "test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5) Run evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# 6) Print results
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))
