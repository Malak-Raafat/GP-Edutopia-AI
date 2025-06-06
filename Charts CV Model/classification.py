import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3
import os
from PIL import Image
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define class names
class_names = ['just_image', 'bar_chart', 'diagram', 'flow_chart', 'graph', 'growth_chart', 'pie_chart', 'table']

# Load trained model
def load_model(model_path, num_classes):
    try:
        model = efficientnet_b3(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Predict image class
def predict_image(image_path, model, transform, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            return class_names[predicted.item()]
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

# Inference function
def infer_and_save(input_folder, output_folder, model_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes=len(class_names)).to(device)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            predicted_class = predict_image(image_path, model, transform, device)
            
            if predicted_class and predicted_class != 'just_image':
                new_filename = f"{predicted_class}_{filename}"
                output_path = os.path.join(output_folder, new_filename)
                shutil.copy(image_path, output_path)
                logging.info(f"Saved: {output_path}")

# Run inference
input_folder = r"D:\data\cropped_objects_local_colab_pretrained_aug"
output_folder = r"D:\data\cropped_objects_local_colab_pretrained_aug\classified"
model_path = r"D:\data\eff.pth"

infer_and_save(input_folder, output_folder, model_path)
