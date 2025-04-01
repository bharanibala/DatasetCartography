import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models as tvmodels
from torch.optim import Adam
import json
import os
from PIL import Image

from torchvision.models import ResNet34_Weights
import logging
import random

class EarthquakeDataset(Dataset):
    def __init__(self, img_dir, transform=None, equalize_classes=True):
        self.img_labels = []
        self.img_dir = img_dir
        self.transform = transform
        self.equalize_classes = equalize_classes

        # Map class names to labels - Disaster Dataset
        self.class_map = {'none': 0, 'mild': 1, 'severe': 1}

        # Map class names to labels - Vet Med Dataset
        #self.class_map = {'negative': 0, 'positive': 1}

    def load_images(self):
        # Dictionary to store images for each class
        class_images = {label: [] for label in self.class_map.values()}
        
        # Populate the class_images dictionary
        for class_name, label in self.class_map.items():
            class_dir = os.path.join(self.img_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                class_images[label].append((img_path, label))
        
        # If equalize_classes is True, sample the same number of images for each class
        if self.equalize_classes:
            min_class_count = min(len(images) for images in class_images.values())
            for label, images in class_images.items():
                self.img_labels.extend(random.sample(images, min_class_count))
        else:
            for images in class_images.values():
                self.img_labels.extend(images)

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'guid': idx,
            'image_path': img_path
        }

# Function to save training dynamics
def save_training_dynamics(dynamics, epoch, output_dir):
    
    #print(dynamics)
    #print('====================================')
    output_file = os.path.join(output_dir, f'dynamics_epoch_{epoch}.jsonl')
    with open(output_file, 'w') as f:
        for item in dynamics:
            f.write(json.dumps(item) + '\n')

image_dir = "/home/b/bharanibala/noisefind/aumrank/aum-master/vgg/disaster/nepal_topk/train"
output_dir = "/home/b/bharanibala/noisefind/aumrank/aum-master/vgg/disaster/nepal_topk/notpretrained/training_dynamics"

num_epochs=5
num_classes=2
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Run the training
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize model (e.g., ResNet)
model = tvmodels.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Prepare dataset and dataloader
dataset = EarthquakeDataset(img_dir=image_dir, transform=transform)
dataset.load_images()
#print(len(dataset))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Set up optimizer
optimizer = Adam(model.parameters(), lr=2e-5)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    total_batches = len(dataloader)
    dynamics = []

    for batch in dataloader:
    # Move images and labels to device
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        epoch_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        guids = batch['guid']
        logits = outputs.detach().cpu()
        img_paths = batch['image_path']

        for guid, logit, gold, img_path in zip(guids, logits, labels.cpu(), img_paths):
            dynamics.append({
                'guid': guid.item() if isinstance(guid, torch.Tensor) else guid,
                f'logits_epoch_{epoch}': logit.tolist(),
                'gold': gold.item() if isinstance(gold, torch.Tensor) else gold,
                'image_path': img_path
            })

    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / total_batches
    
    # Log epoch information
    logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_epoch_loss:.4f}")
    
    # Save training dynamics for this epoch
    save_training_dynamics(dynamics, epoch, output_dir)
    
    # Clear lists to free up memory
    dynamics.clear()

# Save the final model
model_save_path = os.path.join(output_dir, 'final_model')
os.makedirs(model_save_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_save_path, 'model.pth'))
logger.info("Training complete. Model saved.")