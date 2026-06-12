#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from PIL import Image

print("🛫 HIGH-RESOLUTION CLEAR AIRPLANE TRAINING - FIXED")
print("=" * 60)

# Configuration - HIGH RESOLUTION
class Config:
    IMG_HEIGHT = 128  # 4X HIGHER RESOLUTION!
    IMG_WIDTH = 128   # 4X HIGHER RESOLUTION!
    IMG_CHANNELS = 3
    BATCH_SIZE = 8    # Smaller batch for high-res
    EPOCHS = 20       # Reduced for testing
    LEARNING_RATE = 0.0001
    BETA_1 = 0.5
    NOISE_DIM = 256
    CHECKPOINT_DIR = './outputs/trained_models/'
    COMPARISON_DIR = './outputs/clear_comparisons/'

# Create output directories
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.COMPARISON_DIR, exist_ok=True)

print("1. Loading and UPSCALING CIFAR-10 Airplane images...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Filter only airplanes (class 0)
airplane_mask_train = (y_train == 0).flatten()
airplane_mask_test = (y_test == 0).flatten()

x_airplanes_train = x_train[airplane_mask_train]
x_airplanes_test = x_test[airplane_mask_test]

print(f"   Found {len(x_airplanes_train)} training airplane images")

# UPSCALE to 128x128 for CLEARER images
def upscale_images(images, target_size=(128, 128)):
    upscaled = []
    for img in images:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(target_size, Image.LANCZOS)
        img_array = np.array(pil_img)
        upscaled.append(img_array)
    return np.array(upscaled)

print("   Upscaling images to 128x128 resolution...")
x_airplanes_highres = upscale_images(x_airplanes_train)
x_test_highres = upscale_images(x_airplanes_test[:8])

# Normalize
x_airplanes_highres = (x_airplanes_highres.astype('float32') - 127.5) / 127.5
x_test_highres = (x_test_highres.astype('float32') - 127.5) / 127.5

print(f"   Final resolution: {Config.IMG_HEIGHT}x{Config.IMG_WIDTH}")

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices(x_airplanes_highres)
dataset = dataset.shuffle(1000).batch(Config.BATCH_SIZE)

print("2. Building FIXED HIGH-RESOLUTION Generator...")
def build_generator():
    noise_input = tf.keras.layers.Input(shape=(Config.NOISE_DIM,))
    
    # Start with dense layer
    x = tf.keras.layers.Dense(4 * 4 * 1024)(noise_input)  # Start from 4x4
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((4, 4, 1024))(x)
    
    # Progressive upsampling to 128x128
    # 4x4 → 8x8
    x = tf.keras.layers.Conv2DTranspose(512, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 8x8 → 16x16
    x = tf.keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 16x16 → 32x32
    x = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 32x32 → 64x64
    x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 64x64 → 128x128
    x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Final convolution for details
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Output layer - MUST be 128x128x3
    output = tf.keras.layers.Conv2D(Config.IMG_CHANNELS, 3, padding='same', activation='tanh')(x)
    
    model = tf.keras.Model(noise_input, output)
    
    # Verify output shape
    test_output = model(tf.random.normal([1, Config.NOISE_DIM]))
    print(f"   Generator output shape: {test_output.shape}")
    assert test_output.shape[1:] == (Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS), \
        f"Generator output shape mismatch: {test_output.shape[1:]}"
    
    return model

generator = build_generator()
print(f"   Generator: {generator.count_params():,} parameters")

print("3. Building Discriminator...")
def build_discriminator():
    image_input = tf.keras.layers.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS))
    
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(image_input)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(image_input, x)
    
    # Verify input shape compatibility
    test_input = tf.random.normal([1, Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS])
    test_output = model(test_input)
    print(f"   Discriminator accepts: {Config.IMG_HEIGHT}x{Config.IMG_WIDTH}x{Config.IMG_CHANNELS}")
    
    return model

discriminator = build_discriminator()
print(f"   Discriminator: {discriminator.count_params():,} parameters")

# Loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(Config.LEARNING_RATE, Config.BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(Config.LEARNING_RATE, Config.BETA_1)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images):
    batch_size = tf.shape(images)[0]
    random_noise = tf.random.normal([batch_size, Config.NOISE_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(random_noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)
    
    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def generate_clear_comparison(model, epoch, ground_truth_images):
    """Generate high-quality comparison images"""
    
    # Generate predictions
    test_noise = tf.random.normal([4, Config.NOISE_DIM])
    predictions = model(test_noise, training=False)
    
    # Create comparison plot
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    
    for i in range(4):
        # Input Noise
        axes[i, 0].imshow(np.random.rand(128, 128, 3) * 0.3 + 0.35)
        axes[i, 0].set_title('Input\n(Random Noise)', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Ground Truth (Real high-res airplane)
        axes[i, 1].imshow((ground_truth_images[i] + 1) / 2.0)
        axes[i, 1].set_title('Ground Truth\n(Real Airplane)', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Predicted (Generated high-res airplane)
        axes[i, 2].imshow((predictions[i] + 1) / 2.0)
        axes[i, 2].set_title('Predicted\n(Generated)', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'HIGH-RESOLUTION AIRPLANE GENERATION - Epoch {epoch}\n128x128 Resolution', 
                 fontsize=16, y=0.98, fontweight='bold')
    plt.savefig(f'{Config.COMPARISON_DIR}/clear_comparison_epoch_{epoch:02d}.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"💎 Saved HIGH-RESOLUTION comparison for epoch {epoch}")

print("4. Starting FIXED HIGH-RESOLUTION Training...")
# Use first 4 test images as ground truth
ground_truth_samples = x_test_highres[:4]

g_losses = []
d_losses = []

print("🚀 Training with PROPER 128x128 resolution")
print("⏱️  Estimated time: 2-3 hours")

for epoch in range(Config.EPOCHS):
    start = time.time()
    epoch_g_losses = []
    epoch_d_losses = []
    
    for batch_images in tqdm(dataset, desc=f"Epoch {epoch+1}/{Config.EPOCHS}"):
        g_loss, d_loss = train_step(batch_images)
        epoch_g_losses.append(g_loss)
        epoch_d_losses.append(d_loss)
    
    avg_g_loss = np.mean(epoch_g_losses)
    avg_d_loss = np.mean(epoch_d_losses)
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)
    
    # Generate high-res comparisons every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == Config.EPOCHS - 1:
        generate_clear_comparison(generator, epoch + 1, ground_truth_samples)
    
    print(f"Epoch {epoch+1}/{Config.EPOCHS} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f} - Time: {time.time()-start:.2f}s")

print("5. Saving high-resolution model...")
generator.save(f'{Config.CHECKPOINT_DIR}/high_res_airplane_generator.h5')

print("=" * 70)
print("🎉 HIGH-RESOLUTION TRAINING COMPLETED!")
print("💎 Images are now 128x128 resolution (16x more pixels than 32x32!)")
print(f"📁 View your CLEAR results: open {Config.COMPARISON_DIR}")