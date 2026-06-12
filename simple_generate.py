#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

print("🚗 GENERATING CAR IMAGES WITH YOUR AI")
print("=" * 40)

# Check if model exists
model_path = "outputs/trained_models/final_generator.h5"

if not os.path.exists(model_path):
    print("❌ No trained model found. Please train first.")
    print(f"   Looking for: {model_path}")
    exit(1)

print("✅ Loading trained model...")
generator = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

print("🎨 Generating car images...")

# Car is class 1 in CIFAR-10
noise = tf.random.normal([1, 100])
label = tf.constant([[1]])  # 1 = automobile/car

generated_image = generator([noise, label], training=False)

plt.figure(figsize=(6, 6))
plt.imshow((generated_image[0] + 1) / 2.0)
plt.title('YOUR AI-GENERATED CAR 🚗')
plt.axis('off')
plt.show()

print("✨ Your AI created a car image!")
print("💡 Run this script again to see different car variations")