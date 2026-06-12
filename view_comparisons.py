#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

print("🔍 VIEWING COMPARISON RESULTS")
print("=" * 50)

comparison_dir = "outputs/comparisons/"
comparison_files = glob.glob(os.path.join(comparison_dir, "*.png"))

if not comparison_files:
    print("❌ No comparison images found. Run train_airplanes_comparison.py first.")
    exit(1)

print(f"✅ Found {len(comparison_files)} comparison images")

# Show first and last comparison to see progress
first_file = min(comparison_files)
last_file = max(comparison_files)

print(f"\n📊 Showing progress: First vs Last Epoch")
print(f"   First: {os.path.basename(first_file)}")
print(f"   Last:  {os.path.basename(last_file)}")

# Display comparisons
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

first_img = mpimg.imread(first_file)
last_img = mpimg.imread(last_file)

ax1.imshow(first_img)
ax1.set_title(f'FIRST EPOCH - {os.path.basename(first_file)}', fontsize=14, fontweight='bold')
ax1.axis('off')

ax2.imshow(last_img)
ax2.set_title(f'FINAL EPOCH - {os.path.basename(last_file)}', fontsize=14, fontweight='bold')
ax2.axis('off')

plt.tight_layout()
plt.show()

print("\n🎯 Look for improvement in:")
print("   • Predicted images becoming more like Ground Truth")
print("   • Clearer airplane shapes over epochs")
print("   • Better color and detail matching")

print(f"\n💡 View all comparisons: open {comparison_dir}")