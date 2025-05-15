import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Load model and class names
model_data = torch.load("model.pth", map_location=torch.device("cpu"))
class_names = model_data['class_names']

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(model_data['model_state_dict'])
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Streamlit UI
st.set_page_config(page_title="Few-Shot Image Classifier", layout="centered")
st.title("🔍 Few-Shot Image Classification with MAML")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]

    st.success(f"🎯 Predicted Class: {prediction}")
