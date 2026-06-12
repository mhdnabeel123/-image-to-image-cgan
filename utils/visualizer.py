import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from PIL import Image

class Visualizer:
    def __init__(self, config, class_names=None):
        self.config = config
        self.class_names = class_names or [str(i) for i in range(config.N_CLASSES)]
    
    def generate_and_save_images(self, generator, epoch, test_input, test_labels):
        """Generate and save images for given epoch"""
        predictions = generator([test_input, test_labels], training=False)
        
        fig = plt.figure(figsize=(10, 10))
        
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((predictions[i] + 1) / 2.0)  # Scale from [-1,1] to [0,1]
            plt.title(f'Class: {self.class_names[test_labels[i][0]]}')
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if not os.path.exists(self.config.SAMPLE_DIR):
            os.makedirs(self.config.SAMPLE_DIR)
        
        plt.savefig(f'{self.config.SAMPLE_DIR}/image_at_epoch_{epoch:04d}.png')
        plt.close()
    
    def plot_training_history(self, g_losses, d_losses):
        """Plot training history"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses')
        
        plt.subplot(1, 2, 2)
        plt.plot(g_losses, label='Generator Loss', alpha=0.7)
        plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.legend()
        plt.title('Training Losses (Log Scale)')
        
        plt.tight_layout()
        plt.savefig(f'{self.config.SAMPLE_DIR}/training_history.png')
        plt.close()
    
    def display_image_grid(self, images, labels, n_cols=4):
        """Display a grid of images with their labels"""
        n_images = len(images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 3 * n_rows))
        
        for i in range(n_images):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow((images[i] + 1) / 2.0)
            plt.title(f'Class: {self.class_names[labels[i]]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()