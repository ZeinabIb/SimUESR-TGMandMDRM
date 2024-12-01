import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from simUESR_arch import SimUESR

dataset_path_train = "C:\\Users\\Zeinab\\Desktop\\work\\research\\SimUESR-main\\SimUESR-main\\arch\\data_lr_2x\\train"
dataset_path_val = "C:\\Users\\Zeinab\\Desktop\\work\\research\\SimUESR-main\\SimUESR-main\\arch\\data_lr_2x\\val"
# --- Configuration --- #
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset and DataLoader --- #
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# Ensure consistent transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Match the model's target size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# # If needed, interpolate inputs to match output size in loss computation
# loss = criterion(outputs, nn.functional.interpolate(inputs, size=outputs.shape[2:]))

# Replace 'your_dataset_path' with your dataset path.
# train_dataset = datasets.ImageFolder(root='your_dataset_path/train', transform=transform)
train_dataset = datasets.ImageFolder(root=dataset_path_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = datasets.ImageFolder(root=dataset_path_val, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model, Loss, and Optimizer --- #
model = SimUESR(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, 
                n_MDRM=2, height=3, width=2, scale=1, bias=False, task=None)

model = model.to(DEVICE)

criterion = nn.MSELoss()  # Example loss function; adjust as needed
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training and Validation Loops --- #
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, _) in enumerate(train_loader):
        inputs = inputs.to(DEVICE)
        atten = torch.ones((inputs.size(0), 1, inputs.size(2), inputs.size(3))).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs, atten)
        
        # Dynamically adjust input size to match outputs
        inputs_resized = nn.functional.interpolate(inputs, size=outputs.shape[2:])
        loss = criterion(outputs, inputs_resized)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    return running_loss / len(train_loader)


def validate():
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(DEVICE)
            atten = torch.ones((inputs.size(0), 1, inputs.size(2), inputs.size(3))).to(DEVICE)
            outputs = model(inputs, atten)
            
            inputs_resized = nn.functional.interpolate(inputs, size=outputs.shape[2:])
            loss = criterion(outputs, inputs_resized)
            val_loss += loss.item()

    return val_loss / len(val_loader)


# --- Main Training Loop --- #
best_val_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(epoch)
    val_loss = validate()

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model.")

print("Training complete.")
