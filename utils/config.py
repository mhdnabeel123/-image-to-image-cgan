import tensorflow as tf

class Config:
    # Dataset parameters
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3
    N_CLASSES = 10  # Will be updated based on dataset
    
    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.0002
    BETA_1 = 0.5
    
    # Model parameters
    NOISE_DIM = 100
    EMBEDDING_DIM = 50
    FILTERS_GEN = 64
    FILTERS_DISC = 64
    
    # Paths
    CHECKPOINT_DIR = './outputs/trained_models/'
    SAMPLE_DIR = './outputs/generated_images/'
    LOG_DIR = './outputs/training_logs/'
    
    # Training settings
    SAVE_INTERVAL = 10
    SAMPLE_INTERVAL = 5