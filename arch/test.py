import os
import torch
from torchvision import transforms
from PIL import Image
from simUESR_arch import SimUESR

# --- Configuration --- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
INPUT_DIR = "C:\\Users\\Zeinab\\Desktop\\work\\research\\SimUESR-main\\SimUESR-main\\arch\\data_lr_2x\\val\\images\\" 
OUTPUT_DIR = "C:\\Users\\Zeinab\\Desktop\\work\\research\\SimUESR-main\\SimUESR-main\\arch\\output_images\\"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Load Model --- #
model = SimUESR(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_MDRM=2, height=3, width=2, scale=2, bias=False, task=None)
try:
    pretrained_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

model_dict = model.state_dict()
# Filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
# Overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# Load the updated state dict
model.load_state_dict(model_dict)
model = model.to(DEVICE)
model.eval()

# --- Preprocessing --- #
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust based on model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        exit()
    return transform(image).unsqueeze(0)  # Add batch dimension

def postprocess_image(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)  # De-normalize
    image = transforms.ToPILImage()(tensor)
    return image

# --- Process All Images in Directory --- #
for file_name in os.listdir(INPUT_DIR):
    input_image_path = os.path.join(INPUT_DIR, file_name)
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Skipping non-image file: {file_name}")
        continue
    
    print(f"Processing: {file_name}")
    input_tensor = preprocess_image(input_image_path).to(DEVICE)
    
    # Example attention map (adjust or replace as needed)
    attention_map = torch.ones((1, 1, input_tensor.size(2), input_tensor.size(3))).to(DEVICE)
    
    # --- Run Inference --- #
    with torch.no_grad():
        output_tensor = model(input_tensor, attention_map)
    
    # --- Save Output Image --- #
    output_image = postprocess_image(output_tensor.cpu())
    output_image_path = os.path.join(OUTPUT_DIR, f"output_{file_name}")
    try:
        output_image.save(output_image_path)
        print(f"Saved: {output_image_path}")
    except IOError:
        print(f"Error: Unable to save image to {output_image_path}")
