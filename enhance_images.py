import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import glob
import os

def enhance_generated_images():
    """Apply image enhancement to make generated images look clearer"""
    
    image_files = glob.glob("outputs/generated_images/*.png")
    
    for img_path in image_files:
        # Open image
        img = Image.open(img_path)
        
        # Apply enhancements
        img = img.filter(ImageFilter.SHARPEN)
        img = img.filter(ImageFilter.DETAIL)
        
        # Increase contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)
        
        # Save enhanced version
        enhanced_path = img_path.replace('.png', '_enhanced.png')
        img.save(enhanced_path, 'PNG', quality=95)
        
        print(f"Enhanced: {os.path.basename(enhanced_path)}")

enhance_generated_images()
print("✅ Applied image enhancement to all generated images!")