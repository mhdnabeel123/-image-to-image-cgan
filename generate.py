#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Class names from CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def generate_images(class_name=None, num_images=1):
    """Generate images using the trained model"""
    
    # Check if model exists
    model_path = "outputs/trained_models/final_generator.h5"
    
    if not os.path.exists(model_path):
        print("❌ No trained model found. Please train first with: python3 train_fixed.py")
        print(f"   Looking for: {model_path}")
        return
    
    print("✅ Loading trained model...")
    try:
        generator = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Determine class ID
    if class_name:
        # Find class ID from name
        class_name_lower = class_name.lower()
        if class_name_lower in class_names:
            class_id = class_names.index(class_name_lower)
        else:
            # Try to match partial names
            matches = [i for i, name in enumerate(class_names) if class_name_lower in name]
            if matches:
                class_id = matches[0]
                print(f"🔍 Matched '{class_name}' to '{class_names[class_id]}'")
            else:
                print(f"❌ Unknown class: {class_name}")
                print("   Available classes:", ", ".join(class_names))
                return
    else:
        # No class specified, show all
        class_id = None
    
    if class_id is not None:
        # Generate specific class
        print(f"🎨 Generating {num_images} {class_names[class_id]} image(s)...")
        
        if num_images == 1:
            # Single image
            noise = tf.random.normal([1, 100])
            label = tf.constant([[class_id]])
            generated_image = generator([noise, label], training=False)
            
            plt.figure(figsize=(5, 5))
            plt.imshow((generated_image[0] + 1) / 2.0)
            plt.title(f'AI-Generated {class_names[class_id].upper()}')
            plt.axis('off')
            plt.show()
            
        else:
            # Multiple images
            rows = (num_images + 3) // 4
            cols = min(num_images, 4)
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
            if num_images > 1:
                axes = axes.flat
            else:
                axes = [axes]
            
            for i in range(num_images):
                noise = tf.random.normal([1, 100])
                label = tf.constant([[class_id]])
                generated_image = generator([noise, label], training=False)
                
                axes[i].imshow((generated_image[0] + 1) / 2.0)
                axes[i].set_title(f'{class_names[class_id]} #{i+1}')
                axes[i].axis('off')
            
            # Hide empty subplots
            for i in range(num_images, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.suptitle(f'AI-Generated {class_names[class_id].upper()} Variations', y=1.02)
            plt.show()
        
        print(f"✨ Your AI created {num_images} {class_names[class_id]} image(s)!")
    
    else:
        # Show all classes
        print("🖼️  Generating one image for each class...")
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flat
        
        for class_id in range(10):
            noise = tf.random.normal([1, 100])
            label = tf.constant([[class_id]])
            generated_image = generator([noise, label], training=False)
            
            axes[class_id].imshow((generated_image[0] + 1) / 2.0)
            axes[class_id].set_title(f'{class_names[class_id].upper()}')
            axes[class_id].axis('off')
        
        plt.tight_layout()
        plt.suptitle('YOUR AI-GENERATED IMAGES - ALL CLASSES', fontsize=16, y=1.02)
        plt.show()
        
        print("✨ Your AI created images for all 10 classes!")

def show_help():
    """Show usage instructions"""
    print("🚀 IMAGE GENERATION - USAGE:")
    print("=" * 40)
    print("python3 generate.py                    # Show all classes")
    print("python3 generate.py car               # Generate 1 car image")
    print("python3 generate.py cat 4             # Generate 4 cat images")
    print("python3 generate.py bird 9            # Generate 9 bird images")
    print("\n📋 Available classes:")
    for i, name in enumerate(class_names):
        print(f"   {name:12} (use: python3 generate.py {name})")
    print("\n💡 Examples:")
    print("   python3 generate.py dog 3")
    print("   python3 generate.py airplane")
    print("   python3 generate.py horse 6")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - show all classes
        generate_images()
    
    elif len(sys.argv) == 2:
        if sys.argv[1] in ['-h', '--help', 'help']:
            show_help()
        else:
            # One argument - class name
            generate_images(class_name=sys.argv[1])
    
    elif len(sys.argv) == 3:
        # Two arguments - class name and number
        generate_images(class_name=sys.argv[1], num_images=int(sys.argv[2]))
    
    else:
        print("❌ Invalid arguments")
        show_help()
