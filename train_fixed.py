#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow logging

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

print("🚀 Image-to-Image cGAN Training - Fixed Version")
print("=" * 60)

# Configuration
class Config:
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    IMG_CHANNELS = 3
    N_CLASSES = 10
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.0002
    BETA_1 = 0.5
    NOISE_DIM = 100
    EMBEDDING_DIM = 50
    FILTERS_GEN = 64
    FILTERS_DISC = 64
    CHECKPOINT_DIR = './outputs/trained_models/'
    SAMPLE_DIR = './outputs/generated_images/'

# Create output directories
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.SAMPLE_DIR, exist_ok=True)

print("1. Loading CIFAR-10 dataset...")
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(10000).batch(Config.BATCH_SIZE)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f"   Loaded {x_train.shape[0]} images from {len(class_names)} classes")

print("2. Building Generator...")
# Generator
def build_generator():
    noise_input = tf.keras.layers.Input(shape=(Config.NOISE_DIM,))
    label_input = tf.keras.layers.Input(shape=(1,), dtype='int32')
    
    # Label embedding
    label_embed = tf.keras.layers.Embedding(Config.N_CLASSES, Config.EMBEDDING_DIM)(label_input)
    label_embed = tf.keras.layers.Dense(Config.IMG_HEIGHT * Config.IMG_WIDTH)(label_embed)
    label_embed = tf.keras.layers.Reshape((Config.IMG_HEIGHT, Config.IMG_WIDTH, 1))(label_embed)
    
    # Noise processing
    noise_dense = tf.keras.layers.Dense(Config.IMG_HEIGHT * Config.IMG_WIDTH * 64)(noise_input)
    noise_reshape = tf.keras.layers.Reshape((Config.IMG_HEIGHT, Config.IMG_WIDTH, 64))(noise_dense)
    
    # Concatenate
    concatenated = tf.keras.layers.Concatenate()([noise_reshape, label_embed])
    
    # Conv layers
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(concatenated)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    
    # Output
    output = tf.keras.layers.Conv2D(Config.IMG_CHANNELS, 3, padding='same', activation='tanh')(x)
    
    return tf.keras.Model([noise_input, label_input], output)

generator = build_generator()
print(f"   Generator built: {generator.count_params():,} parameters")

print("3. Building Discriminator...")
def build_discriminator():
    image_input = tf.keras.layers.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS))
    label_input = tf.keras.layers.Input(shape=(1,), dtype='int32')
    
    # Label embedding
    label_embed = tf.keras.layers.Embedding(Config.N_CLASSES, Config.EMBEDDING_DIM)(label_input)
    label_embed = tf.keras.layers.Dense(Config.IMG_HEIGHT * Config.IMG_WIDTH)(label_embed)
    label_embed = tf.keras.layers.Reshape((Config.IMG_HEIGHT, Config.IMG_WIDTH, 1))(label_embed)
    
    # Concatenate
    concatenated = tf.keras.layers.Concatenate()([image_input, label_embed])
    
    # Conv layers
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(concatenated)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model([image_input, label_input], output)

discriminator = build_discriminator()
print(f"   Discriminator built: {discriminator.count_params():,} parameters")

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
def train_step(images, labels):
    batch_size = tf.shape(images)[0]
    
    # Generate random noise
    random_noise = tf.random.normal([batch_size, Config.NOISE_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        generated_images = generator([random_noise, labels], training=True)
        
        # Discriminator outputs
        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)
        
        # Calculate losses
        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)
    
    # Calculate gradients
    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input, test_labels):
    predictions = model([test_input, test_labels], training=False)
    
    fig = plt.figure(figsize=(10, 10))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2.0)
        plt.title(f'Class: {class_names[test_labels[i][0]]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{Config.SAMPLE_DIR}/image_at_epoch_{epoch:04d}.png')
    plt.close()

print("4. Starting training...")
# Create sample noise for visualization
sample_noise = tf.random.normal([16, Config.NOISE_DIM])
sample_labels = tf.constant([[i % Config.N_CLASSES] for i in range(16)])

g_losses = []
d_losses = []

for epoch in range(Config.EPOCHS):
    start = time.time()
    
    epoch_g_losses = []
    epoch_d_losses = []
    
    for batch_images, batch_labels in tqdm(dataset, desc=f"Epoch {epoch+1}/{Config.EPOCHS}"):
        g_loss, d_loss = train_step(batch_images, batch_labels)
        epoch_g_losses.append(g_loss)
        epoch_d_losses.append(d_loss)
    
    # Calculate epoch averages
    avg_g_loss = np.mean(epoch_g_losses)
    avg_d_loss = np.mean(epoch_d_losses)
    
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)
    
    # Generate sample images
    generate_and_save_images(generator, epoch + 1, sample_noise, sample_labels)
    
    print(f"Epoch {epoch+1}/{Config.EPOCHS} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f} - Time: {time.time()-start:.2f}s")

print("5. Saving final models...")
generator.save(f'{Config.CHECKPOINT_DIR}/final_generator.h5')
discriminator.save(f'{Config.CHECKPOINT_DIR}/final_discriminator.h5')

print("6. Plotting training history...")
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.savefig(f'{Config.SAMPLE_DIR}/training_history.png')
plt.show()

print("=" * 60)
print("🎉 Training completed successfully!")
print(f"📁 Generated images saved in: {Config.SAMPLE_DIR}")
print(f"📁 Models saved in: {Config.CHECKPOINT_DIR}")
print("\nYou can now generate new images with: python3 generate.py")