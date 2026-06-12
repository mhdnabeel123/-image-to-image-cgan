#!/usr/bin/env python3
import os

def check_structure():
    print("📁 Checking Project Structure...")
    print("=" * 50)
    
    structure = {
        'models/': ['__init__.py', 'generator.py', 'discriminator.py', 'cgan.py'],
        'utils/': ['__init__.py', 'config.py', 'data_loader.py', 'losses.py', 'visualizer.py'],
        'training/': ['__init__.py', 'trainer.py'],
        '': ['train.py', 'generate.py', 'requirements.txt']
    }
    
    all_good = True
    
    for folder, files in structure.items():
        folder_path = folder if folder else '.'
        
        if not os.path.exists(folder_path):
            print(f"❌ Missing folder: {folder_path}")
            all_good = False
            continue
            
        print(f"📁 {folder_path}/")
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.exists(file_path):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} (MISSING)")
                all_good = False
    
    print("=" * 50)
    if all_good:
        print("🎉 Project structure is complete!")
    else:
        print("❌ Some files are missing. Please check the structure.")
    
    return all_good

if __name__ == "__main__":
    check_structure()