import tensorflow as tf
from tensorflow.keras import layers, Model

class Discriminator:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()
    
    def build_model(self):
        # Input layers - image and condition
        input_image = layers.Input(shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS))
        input_label = layers.Input(shape=(1,), dtype='int32')
        
        # Label embedding
        label_embedding = layers.Embedding(self.config.N_CLASSES, self.config.EMBEDDING_DIM)(input_label)
        label_embedding = layers.Dense(self.config.IMG_HEIGHT * self.config.IMG_WIDTH)(label_embedding)
        label_embedding = layers.Reshape((self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 1))(label_embedding)
        
        # Concatenate image and label information
        concatenated = layers.Concatenate()([input_image, label_embedding])
        
        # PatchGAN discriminator
        x = layers.Conv2D(self.config.FILTERS_DISC, kernel_size=4, strides=2, padding='same')(concatenated)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(self.config.FILTERS_DISC * 2, kernel_size=4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(self.config.FILTERS_DISC * 4, kernel_size=4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(self.config.FILTERS_DISC * 8, kernel_size=4, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Final classification layer
        x = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(x)
        validity = layers.Activation('sigmoid')(x)
        
        model = Model(inputs=[input_image, input_label], outputs=validity)
        return model