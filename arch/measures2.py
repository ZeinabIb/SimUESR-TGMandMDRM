import os
import cv2
import numpy as np
from math import log10, sqrt

def PSNR(original, compressed):
    if original is None or compressed is None:
        raise ValueError("One or both images could not be loaded.")
    
    # Ensure the dimensions are the same
    if original.shape != compressed.shape:
        raise ValueError(f"Image dimensions do not match: {original.shape} vs {compressed.shape}")
    
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
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
    I1 = np.float32(I1)
    I2 = np.float32(I2)
    
    mu_X = np.mean(I1)
    mu_Y = np.mean(I2)
    
    sigma_X_sq = np.var(I1)
    sigma_Y_sq = np.var(I2)
    
    sigma_XY = np.cov(I1.flatten(), I2.flatten())[0][1]
    
    uqi = (4 * mu_X * mu_Y * sigma_XY) / ((mu_X**2 + mu_Y**2) * (sigma_X_sq + sigma_Y_sq))
    
    return uqi

def main():
    # Directories
    original_dir = "C:\\Users\\Zeinab\\Desktop\\work\\research\\SimUESR-main\\SimUESR-main\\arch\\data_lr_2x\\val\\images"
    compressed_dir = "C:\\Users\\Zeinab\\Desktop\\work\\research\\SimUESR-main\\SimUESR-main\\arch\\output_images"
    
    # Get sorted list of image files in both directories
    original_images = sorted([f for f in os.listdir(original_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    compressed_images = sorted([f for f in os.listdir(compressed_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    # Ensure both directories have the same number of images
    if len(original_images) != len(compressed_images):
        print(f"Warning: The number of images in {original_dir} and {compressed_dir} don't match.")
        return

    # Process each pair of images
    for original_img, compressed_img in zip(original_images, compressed_images):
        original_path = os.path.join(original_dir, original_img)
        compressed_path = os.path.join(compressed_dir, compressed_img)
        
        # Check if the compressed image exists
        if not os.path.exists(compressed_path):
            print(f"Compressed image for {original_img} not found at {compressed_path}")
            continue
        
        try:
            # Read original and compressed images
            original = cv2.imread(original_path)
            compressed = cv2.imread(compressed_path)

            # Resize the compressed image to match the original image dimensions
            compressed_resized = cv2.resize(compressed, (original.shape[1], original.shape[0]))

            # Calculate PSNR
            psnr_value = PSNR(original, compressed_resized)
            print(f"PSNR for {original_img}: {psnr_value} dB")

            # Convert to grayscale for SSIM and UIQM
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            compressed_gray = cv2.cvtColor(compressed_resized, cv2.COLOR_BGR2GRAY)

            # Calculate SSIM
            ssim_value = getSSISM(original_gray, compressed_gray)
            print(f"SSIM for {original_img}: {ssim_value}")

            # Calculate UIQM
            uqi_value = getUQI(original_gray, compressed_gray)
            print(f"UIQM for {original_img}: {uqi_value}")

        except Exception as e:
            print(f"Error processing {original_img}: {e}")

if __name__ == "__main__":
    main()
