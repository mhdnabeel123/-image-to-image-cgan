import tensorflow as tf
from tensorflow.keras import Model

class ConditionalGAN:
    def __init__(self, config):
        self.config = config
        from .generator import Generator
        from .discriminator import Discriminator
        
        self.generator = Generator(config).model
        self.discriminator = Discriminator(config).model
        
        # Build the complete CGAN model
        self.build_complete_model()
        
    def build_complete_model(self):
        # Generator inputs
        noise = tf.keras.Input(shape=(self.config.NOISE_DIM,))
        label = tf.keras.Input(shape=(1,), dtype='int32')
        
        # Generated image
        generated_image = self.generator([noise, label])
        
        # For the combined model, we only train the generator
        self.discriminator.trainable = False
        
        # Discriminator determines validity of generated images
        validity = self.discriminator([generated_image, label])
        
        # Combined model (stacked generator and discriminator)
        self.combined = Model([noise, label], validity)
        
    def get_models(self):
        return self.generator, self.discriminator, self.combined