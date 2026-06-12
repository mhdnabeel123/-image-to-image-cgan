import tensorflow as tf

class CGANLoss:
    def __init__(self):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    def discriminator_loss(self, real_output, fake_output):
        """
        Discriminator loss function
        real_output: discriminator output for real images
        fake_output: discriminator output for fake images
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_disc_loss = real_loss + fake_loss
        return total_disc_loss
    
    def generator_loss(self, fake_output):
        """
        Generator loss function
        fake_output: discriminator output for fake images
        """
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def l1_loss(self, generated_images, target_images):
        """L1 loss for image-to-image translation"""
        return tf.reduce_mean(tf.abs(generated_images - target_images))