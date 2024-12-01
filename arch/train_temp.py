import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from simUESR_arch import SimUESR

dataset_path = "C:\\Users\\Zeinab\\Desktop\\work\\research\\SimUESR-main\\SimUESR-main\\arch\\data_lr_2x"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

print("Dataset path:", dataset_path)
print("Contents:", os.listdir(dataset_path))

# --- Hyperparameters --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

learning_rate = 1e-4
batch_size = 8
num_epochs = 100
scale = 2
task = None  # Set 'defocus_deblurring' for that task, or None for super-resolution
log_interval = 10
import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
# --- Data Transformations --- #
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = CustomDataset(root=dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print(f"DataLoader initialized with batch size: {batch_size}")

# --- Model, Loss, Optimizer --- #
print("Initializing model...")
model = SimUESR(
    inp_channels=3,
    out_channels=3,
    n_feat=80,
    chan_factor=1.5,
    n_MDRM=2,
    height=3,
    width=2,
    scale=scale,
    bias=False,
    task=task
).to(device)
print(f"Model initialized on device: {device}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("Loss and optimizer initialized.")

# --- Training Loop --- #
def train_model():
    print("Starting training loop...")
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        # for batch_idx, (inputs, attentions) in enumerate(progress_bar):
        #     print(f"Processing batch {batch_idx + 1}")
        #     inputs, attentions = inputs.to(device), attentions.to(device)
        #     print(f"Inputs shape: {inputs.shape}, Attentions shape: {attentions.shape}")
        for batch_idx, inputs in enumerate(progress_bar):
            print(f"Processing batch {batch_idx + 1}")
            inputs = inputs.to(device)
            print(f"Inputs shape: {inputs.shape}")


            # Forward pass
            # outputs = model(inputs, attentions)
            outputs = model(inputs)

            print(f"Outputs shape: {outputs.shape}")
            loss = criterion(outputs, inputs)
            print(f"Loss: {loss.item()}")

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % log_interval == 0:
                print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")
                progress_bar.set_postfix({"Batch Loss": loss.item()})

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

# --- Save Model --- #
def save_model(model, path="simuesr_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    train_model()
    save_model(model)
