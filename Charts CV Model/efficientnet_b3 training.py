import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Define the dataset class (already defined in your code)
class GraphsDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, file_list, transform=None):
        self.dir_path = dir_path
        self.file_list = file_list
        self.transform = transform
        self.label_dict = {
            'just_image': 0, 
            'bar_chart': 1, 
            'diagram': 2, 
            'flow_chart': 3, 
            'graph': 4, 
            'growth_chart': 5,
            'pie_chart': 6, 
            'table': 7
        }
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.dir_path, self.file_list[idx])
        
        # Handle GIF files differently
        if image_name.split('.')[::-1][0] == "gif":
            gif = cv2.VideoCapture(image_name)
            _, image = gif.read()
        else:
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract label from filename
        label = None
        for name, label_id in self.label_dict.items():
            if name in image_name.split('/')[::-1][0]:
                label = label_id
                break
                
        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label

# Set up paths and list files
dataset_dir = '/kaggle/input/graphs-dataset/graphs'        
dataset_files = os.listdir(dataset_dir)
class_names = ['just_image', 'bar_chart', 'diagram', 'flow_chart', 'graph', 'growth_chart', 'pie_chart', 'table']
print(f"Total images: {len(dataset_files)}")

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create dataset
full_dataset = GraphsDataset(dataset_dir, dataset_files, transform)

# Split into train, validation, and test sets
train_size = int(len(full_dataset) * 0.7)  # 70% for training
val_size = int(len(full_dataset) * 0.15)   # 15% for validation
test_size = len(full_dataset) - train_size - val_size  # 15% for testing

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

# Create data loaders
batch_size = 32  # Smaller batch size to avoid memory issues
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=2  # Parallel loading
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

# Print dataset information
print(f"Training set: {len(train_dataset)} samples")
print(f"Validation set: {len(val_dataset)} samples")
print(f"Test set: {len(test_dataset)} samples")
print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")

# Load MobileNetV3-Large model
model = torchvision.models.efficientnet_b3(pretrained=True)

# Modify the final classification layer
num_classes = len(class_names)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# Training loop
num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Train
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step()
    
    # Print results
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model!")

# Load the best model for testing
model.load_state_dict(torch.load("best_model.pth"))

# Test function
def test(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # Collect predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(conf_matrix)

# Evaluate on the test set
print("Testing the model on the test set...")
test(model, test_loader, device)