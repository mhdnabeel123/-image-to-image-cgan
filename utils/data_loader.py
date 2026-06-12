import tensorflow as tf
import numpy as np
import os
from PIL import Image

class DataLoader:
    def __init__(self, config):
        self.config = config
    
    def load_cifar10(self):
        """Load CIFAR-10 dataset for conditional generation"""
        (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
        
        # Normalize images to [-1, 1]
        x_train = (x_train.astype('float32') - 127.5) / 127.5
        
        # Update config with dataset info
        self.config.N_CLASSES = 10
        self.config.IMG_HEIGHT = 32
        self.config.IMG_WIDTH = 32
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(buffer_size=10000).batch(self.config.BATCH_SIZE)
        
        return dataset
    
    def load_image_to_image_dataset(self, input_dir, target_dir):
        """
        Load image-to-image translation dataset
        input_dir: directory with input images
        target_dir: directory with target images
        """
        input_images = []
        target_images = []
        
        # Get sorted list of image files
        input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        target_files = sorted([f for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for inp_file, tar_file in zip(input_files, target_files):
            # Load input image
            inp_path = os.path.join(input_dir, inp_file)
            inp_img = Image.open(inp_path).convert('RGB')
            inp_img = inp_img.resize((self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            inp_array = np.array(inp_img)
            inp_array = (inp_array.astype('float32') - 127.5) / 127.5
            
            # Load target image
            tar_path = os.path.join(target_dir, tar_file)
            tar_img = Image.open(tar_path).convert('RGB')
            tar_img = tar_img.resize((self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            tar_array = np.array(tar_img)
            tar_array = (tar_array.astype('float32') - 127.5) / 127.5
            
            input_images.append(inp_array)
            target_images.append(tar_array)
        
        input_images = np.array(input_images)
        target_images = np.array(target_images)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((input_images, target_images))
        dataset = dataset.shuffle(buffer_size=len(input_images)).batch(self.config.BATCH_SIZE)
        
        return dataset