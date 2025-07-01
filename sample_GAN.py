import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Configuration
IMG_HEIGHT = 256
IMG_WIDTH = 256
CHANNELS = 3
BATCH_SIZE = 1
TAMPER_CATEGORIES = ['text_alteration', 'signature_forgery', 'photo_swap', 
                   'print_scan_artifacts', 'partial_erasure']
NUM_SAMPLES = 500

# Generator (U-Net based)
def build_generator():
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, CHANNELS])
    
    # Downsampling
    down_stack = [
        layers.Conv2D(64, 4, strides=2, padding='same', activation='leaky_relu'),
        layers.Conv2D(128, 4, strides=2, padding='same', activation='leaky_relu'),
        layers.Conv2D(256, 4, strides=2, padding='same', activation='leaky_relu'),
        layers.Conv2D(512, 4, strides=2, padding='same', activation='leaky_relu'),
        layers.Conv2D(512, 4, strides=2, padding='same', activation='leaky_relu'),
    ]
    
    # Upsampling
    up_stack = [
        layers.Conv2DTranspose(512, 4, strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(256, 4, strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu'),
    ]
    
    # Last layer
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(CHANNELS, 4, strides=2, padding='same',
                                activation='tanh', kernel_initializer=initializer)
    
    x = inputs
    
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    
    x = last(x)
    
    return Model(inputs=inputs, outputs=x)

# Discriminator (PatchGAN)
def build_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, CHANNELS], name='input_image')
    
    x = layers.Conv2D(64, 4, strides=2, padding='same', 
                     kernel_initializer=initializer)(inp)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, 4, strides=2, padding='same', 
                     kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(256, 4, strides=2, padding='same', 
                     kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(512, 4, strides=1, padding='same', 
                     kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    last = layers.Conv2D(1, 4, strides=1, padding='same', 
                        kernel_initializer=initializer)(x)
    
    return Model(inputs=inp, outputs=last)

# Category-Specific Tampering Generator
class DocumentTamperGAN:
    def __init__(self):
        self.generators = {category: build_generator() for category in TAMPER_CATEGORIES}
        self.discriminators = {category: build_discriminator() for category in TAMPER_CATEGORIES}
        
        # Loss functions
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.LAMBDA = 10  # Cycle consistency weight
        
        # Optimizers
        self.generator_optimizers = {
            category: tf.keras.optimizers.Adam(2e-4, beta_1=0.5) 
            for category in TAMPER_CATEGORIES
        }
        self.discriminator_optimizers = {
            category: tf.keras.optimizers.Adam(2e-4, beta_1=0.5) 
            for category in TAMPER_CATEGORIES
        }
    
    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        return real_loss + generated_loss
    
    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)
    
    def calc_cycle_loss(self, real_image, cycled_image):
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss
    
    def generate_tampered_samples(self, original_image, category, num_samples):
        # Preprocess input
        original_image = cv2.resize(original_image, (IMG_WIDTH, IMG_HEIGHT))
        original_image = (original_image / 127.5) - 1.0
        original_image = np.expand_dims(original_image, axis=0)
        
        # Generate samples
        generated_samples = []
        for _ in range(num_samples):
            # Add random noise to create variations
            noise = np.random.normal(0, 0.1, (1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
            noisy_input = original_image + noise
            
            # Generate tampered version
            generated = self.generators[category](noisy_input, training=True)
            generated_samples.append(generated[0].numpy())
        
        return generated_samples
    
    def postprocess_samples(self, generated_samples, category):
        processed_samples = []
        for sample in generated_samples:
            # Convert from [-1, 1] to [0, 255]
            sample = ((sample + 1) * 127.5).astype(np.uint8)
            
            # Category-specific post-processing
            if category == 'text_alteration':
                # Add visible text alterations
                sample = cv2.putText(sample, "MODIFIED", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            elif category == 'signature_forgery':
                # Add forged signature marker
                cv2.rectangle(sample, (80, 150), (200, 180), (255,0,0), 2)
            elif category == 'print_scan_artifacts':
                # Add print-scan noise
                sample = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
                sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
            
            processed_samples.append(sample)
        
        return processed_samples

# Usage Example
if __name__ == "__main__":
    # Initialize GAN
    tamper_gan = DocumentTamperGAN()
    
    # Load original PAN card (replace with your image path)
    original_pan = cv2.imread("original_pan.jpg")
    original_pan = cv2.cvtColor(original_pan, cv2.COLOR_BGR2RGB)
    
    # Generate samples for each category
    output_dir = "tampered_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    samples_per_category = NUM_SAMPLES // len(TAMPER_CATEGORIES)
    
    for category in TAMPER_CATEGORIES:
        # Generate raw samples
        raw_samples = tamper_gan.generate_tampered_samples(
            original_pan, category, samples_per_category
        )
        
        # Post-process for realism
        final_samples = tamper_gan.postprocess_samples(raw_samples, category)
        
        # Save samples
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for i, sample in enumerate(final_samples):
            cv2.imwrite(
                os.path.join(category_dir, f"tampered_{category}_{i}.png"),
                cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            )
        
        print(f"Generated {len(final_samples)} samples for {category}")
    
    print(f"Total {NUM_SAMPLES} tampered samples generated in {output_dir}")
