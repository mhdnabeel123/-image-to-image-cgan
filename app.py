#!/usr/bin/env python3
"""
Flask backend for the cGAN Studio frontend.

Model routing:
  class 0 (airplane) → improved_airplane_generator.h5  [64×64, noise_dim=256]
  all others          → final_generator.h5              [32×32, noise+label]
"""
import os, io, base64, glob
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, jsonify, send_from_directory, request
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_folder='frontend', static_url_path='')

# ── CIFAR-10 class info ────────────────────────────────────────────────────────
CLASS_NAMES  = ['airplane','automobile','bird','cat','deer',
                'dog','frog','horse','ship','truck']
CLASS_EMOJIS = ['✈️','🚗','🐦','🐱','🦌','🐶','🐸','🐴','🚢','🚚']

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_GENERAL  = 'outputs/trained_models/final_generator.h5'      # 32×32, needs [noise,label]
MODEL_AIRPLANE = 'outputs/trained_models/improved_airplane_generator.h5'  # 64×64, noise only

_models = {}   # lazy cache

def get_model(key):
    if key not in _models:
        path = MODEL_AIRPLANE if key == 'airplane' else MODEL_GENERAL
        print(f"⏳  Loading {key} model from {path}…")
        _models[key] = tf.keras.models.load_model(path)
        print(f"✅  {key} model ready.")
    return _models[key]

# ── Image post-processing ──────────────────────────────────────────────────────
def enhance(pil_img, native_size):
    """
    Clean upscale + balanced sharpening.
    native_size: actual pixel dimension of the raw model output (32 or 64).
    """
    target = 512

    # Step 1 – Upscale with LANCZOS (best quality, no blocking)
    pil_img = pil_img.resize((target, target), Image.LANCZOS)

    # Step 2 – Single moderate UnsharpMask
    #   radius lower = finer detail, percent moderate = no halos
    radius  = 1.2 if native_size >= 64 else 1.8
    percent = 100 if native_size >= 64 else 160
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=radius,
                                                      percent=percent,
                                                      threshold=2))

    # Step 3 – Sharpness (1.0 = original, 2.0 = twice as sharp)
    factor = 1.4 if native_size >= 64 else 1.8
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(factor)

    # Step 4 – Contrast
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.25)

    # Step 5 – Color
    pil_img = ImageEnhance.Color(pil_img).enhance(1.2)

    return pil_img

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/api/classes')
def get_classes():
    return jsonify([
        {'id': i, 'name': CLASS_NAMES[i], 'emoji': CLASS_EMOJIS[i],
         'hires': (i == 0)}          # flag so the UI can show a ✨ badge
        for i in range(len(CLASS_NAMES))
    ])

@app.route('/api/generate', methods=['POST'])
def generate():
    data      = request.get_json(force=True)
    class_id  = int(data.get('class_id', 1))
    num_imgs  = max(1, min(int(data.get('num_images', 1)), 9))

    use_hires = (class_id == 0)          # airplane → high-res model
    model_key = 'airplane' if use_hires else 'general'
    model     = get_model(model_key)

    results = []
    for _ in range(num_imgs):
        if use_hires:
            # 64×64 model: noise dim = 256, no label
            noise      = tf.random.normal([1, 256])
            img_tensor = model(noise, training=False)
            native     = 64
        else:
            # 32×32 model: noise dim = 100 + class label
            noise      = tf.random.normal([1, 100])
            label      = tf.constant([[class_id]])
            img_tensor = model([noise, label], training=False)
            native     = 32

        img_np  = ((img_tensor[0].numpy() + 1) / 2.0 * 255).clip(0, 255).astype('uint8')
        pil_img = Image.fromarray(img_np)
        pil_img = enhance(pil_img, native)

        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        results.append('data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode())

    return jsonify({
        'images':       results,
        'class_name':   CLASS_NAMES[class_id],
        'class_emoji':  CLASS_EMOJIS[class_id],
        'hires':        use_hires,
        'native_px':    native,
        'timestamp':    datetime.now().isoformat()
    })

@app.route('/api/gallery')
def gallery():
    img_dir = 'outputs/generated_images'
    files   = sorted(glob.glob(os.path.join(img_dir, 'samples_epoch_*.png')))
    return jsonify([{
        'path':  f'/outputs/generated_images/{os.path.basename(f)}',
        'label': f'Epoch {int(os.path.basename(f).replace("samples_epoch_","").replace(".png",""))}'
    } for f in files])

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory('outputs', filename)

if __name__ == '__main__':
    print("🌐  cGAN Studio → http://127.0.0.1:5000")
    print("    ✈️  Airplane uses 64×64 high-res model")
    print("    🚗  All other classes use 32×32 conditional model")
    app.run(debug=False, port=5000)
