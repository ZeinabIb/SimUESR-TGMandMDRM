import shutil
import cv2
import numpy as np
import os
import os

original_folder = "C:\\Users\\Zeinab\\Desktop\\work\\research\\SimUESR-main\\SimUESR-main\\arch\\data_lr_2x\\val\\images\\" 
result_folder = "C:\\Users\\Zeinab\\Desktop\\work\\research\\SimUESR-main\\SimUESR-main\\arch\\output_images\\"

original_files = os.listdir(original_folder)
result_files = os.listdir(result_folder)

print("Original Images:", original_files)
print("Result Images:", result_files)

destination_folder = "matched_images"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Get the list of original and result files
original_files = os.listdir(original_folder)
result_files = os.listdir(result_folder)

# Create a dictionary mapping result images without the 'output_' prefix
result_images = {f.replace("output_", ""): f for f in result_files}

# Match files and copy the matched results
for original_file in original_files:
    if original_file in result_images:
        matched_file = result_images[original_file]
        shutil.copy(
            os.path.join(result_folder, matched_file),
            os.path.join(destination_folder, matched_file),
        )
        print(f"Matched and copied: {matched_file}")
    else:
        print(f"Result image for {original_file} not found.")

def getPSNR(I1, I2):
    s1 = cv2.absdiff(I1, I2)
    s1 = np.float32(s1)
    s1 = s1 * s1
    sse = s1.sum()
    if sse <= 1e-10:
        return 0
    else:
        if len(I1.shape) == 2:  # Grayscale image
            mse = sse / (I1.shape[0] * I1.shape[1])
        else:  # Color image
            mse = sse / (I1.shape[0] * I1.shape[1] * I1.shape[2])

        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr

def getSSISM(I1, I2):
    C1 = 6.5025
    C2 = 58.5225

    I1 = np.float32(I1)
    I2 = np.float32(I2)
    I2_2 = I2 * I2
    I1_2 = I1 * I1
    I1_I2 = I1 * I2

    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5) - mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5) - mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5) - mu1_mu2

    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    ssim_map = cv2.divide(t3, t1 * t2)
    ssim = cv2.mean(ssim_map)
    return ssim[0]

def getUQI(I1, I2):
    # Ensure I1 and I2 are float32
    I1 = np.float32(I1)
    I2 = np.float32(I2)
    
    # Compute means
    mu_X = np.mean(I1)
    mu_Y = np.mean(I2)
    
    # Compute variances
    sigma_X_sq = np.var(I1)
    sigma_Y_sq = np.var(I2)
    
    # Compute covariance
    sigma_XY = np.cov(I1.flatten(), I2.flatten())[0][1]
    
    # Compute UQI
    uqi = (4 * mu_X * mu_Y * sigma_XY) / ((mu_X**2 + mu_Y**2) * (sigma_X_sq + sigma_Y_sq))
    
    return uqi
def process_images_in_folder(original_folder, result_folder):
    # Get lists of image files in both folders
    original_images = {os.path.splitext(f)[0]: f for f in os.listdir(original_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    result_images = {os.path.splitext(f)[0]: f for f in os.listdir(result_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    
    if not original_images:
        print("No images found in the original folder.")
        return
    
    total_psnr = 0
    total_ssim = 0
    total_uqi = 0
    num_pairs = 0

    for img_name, original_img_file in original_images.items():
        if img_name not in result_images:
            print(f"Result image for {img_name} not found.")
            continue

        original_img_path = os.path.join(original_folder, original_img_file)
        result_img_path = os.path.join(result_folder, result_images[img_name])

        img1 = cv2.imread(original_img_path)
        img2 = cv2.imread(result_img_path)

        if img1 is None or img2 is None:
            print(f"Could not read images: {original_img_path} or {result_img_path}")
            continue

        if img1.shape != img2.shape:
            print(f"Image size mismatch between {original_img_path} and {result_img_path}")
            continue

        if len(img1.shape) == 3:  # Convert color images to grayscale for SSIM, UQI
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        psnr = getPSNR(img1, img2)
        ssim = getSSISM(img1, img2)
        uqi = getUQI(img1, img2)

        print(f"PSNR between {img_name}: {psnr}")
        print(f"SSIM between {img_name}: {ssim}")
        print(f"UQI between {img_name}: {uqi}")

        total_psnr += psnr
        total_ssim += ssim
        total_uqi += uqi
        num_pairs += 1

    if num_pairs > 0:
        mean_psnr = total_psnr / num_pairs
        mean_ssim = total_ssim / num_pairs
        mean_uqi = total_uqi / num_pairs
        print(f"\nMean PSNR: {mean_psnr}")
        print(f"Mean SSIM: {mean_ssim}")
        print(f"Mean UQI: {mean_uqi}")
    else:
        print("No valid image pairs found.")




original_folder = "C:\\Users\\Zeinab\\Desktop\\work\\research\\SimUESR-main\\SimUESR-main\\arch\\data_lr_2x\\val\\images\\" 
result_folder = "C:\\Users\\Zeinab\\Desktop\\work\\research\\SimUESR-main\\SimUESR-main\\arch\\output_images\\"
process_images_in_folder(original_folder, result_folder)