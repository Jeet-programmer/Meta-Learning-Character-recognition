import os
import shutil
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Config
original_data_dir = Path("datasets")
temp_data_dir = Path("all_data")
batch_size = 32
num_epochs = 5
model_path = "model.pth"

# STEP 1: Flatten and rename folders to include language name
def prepare_flat_dataset():
    if temp_data_dir.exists():
        shutil.rmtree(temp_data_dir)
    temp_data_dir.mkdir()

    for lang_folder in original_data_dir.iterdir():
        if lang_folder.is_dir():
            for class_folder in lang_folder.iterdir():
                if class_folder.is_dir():
                    # Combined label e.g., Bengali_character01
                    new_class_name = f"{lang_folder.name}_{class_folder.name}"
                    target_folder = temp_data_dir / new_class_name
                    target_folder.mkdir(exist_ok=True)
                    for img_file in class_folder.glob("*"):
                        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            shutil.copy(img_file, target_folder / img_file.name)

prepare_flat_dataset()

# STEP 2: Load the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=str(temp_data_dir), transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
class_names = dataset.classes

print("Classes found:", class_names)

# STEP 3: Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# STEP 4: Training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

# STEP 5: Save model and classes
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}, model_path)

print("✅ Training complete. Model saved to model.pth")