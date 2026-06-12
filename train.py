import os
import tensorflow as tf
from utils.config import Config
from utils.data_loader import DataLoader
from utils.visualizer import Visualizer
from models.cgan import ConditionalGAN
from training.trainer import CGANTrainer

def main():
    # Initialize configuration
    config = Config()
    
    # Create output directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Load dataset
    data_loader = DataLoader(config)
    
    print("Choose dataset type:")
    print("1. CIFAR-10 (Conditional Generation)")
    print("2. Image-to-Image Translation")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        dataset = data_loader.load_cifar10()
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif choice == "2":
        input_dir = input("Enter input images directory: ")
        target_dir = input("Enter target images directory: ")
        dataset = data_loader.load_image_to_image_dataset(input_dir, target_dir)
        class_names = ['input', 'target']
    else:
        print("Invalid choice. Using CIFAR-10.")
        dataset = data_loader.load_cifar10()
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Initialize visualizer
    visualizer = Visualizer(config, class_names)
    
    # Build CGAN model
    print("Building CGAN model...")
    cgan = ConditionalGAN(config)
    generator, discriminator, combined = cgan.get_models()
    
    # Print model summaries
    print("\nGenerator Summary:")
    generator.summary()
    print("\nDiscriminator Summary:")
    discriminator.summary()
    
    # Initialize trainer
    trainer = CGANTrainer(generator, discriminator, combined, config, visualizer)
    
    # Start training
    epochs = int(input(f"Enter number of epochs (default: {config.EPOCHS}): ") or config.EPOCHS)
    trainer.train(dataset, epochs)
    
    print("Training completed!")

if __name__ == "__main__":
    main()