#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

print("✈️ GENERATING AIRPLANES (5-epoch model)")
print("=" * 40)

model_path = "outputs/trained_models/airplane_quick.h5"

if not os.path.exists(model_path):
    print("❌ No quick airplane model found. Run train_airplanes_quick.py first")
    exit(1)

print("✅ Loading quick airplane model...")
generator = tf.keras.models.load_model(model_path)

print("🎨 Generating 9 airplane images...")
noise = tf.random.normal([9, 100])
airplanes = generator(noise, training=False)

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow((airplanes[i] + 1) / 2.0)
    plt.axis('off')

plt.suptitle('QUICK AI-GENERATED AIRPLANES (5 Epochs)', fontsize=16, y=0.95)
plt.tight_layout()
plt.show()

print("✨ Generated 9 airplane images from 5-epoch training!")