'''
Reference: https://www.geeksforgeeks.org/generative-adversarial-network-gan/
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from gan.discriminator_v1 import Discriminator
from gan.generator_v1 import Generator

# Set device
device = 'cuda' if tf.test.is_gpu_available() else 'cpu'

# Define a basic transform for the images
transform = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255),
])

(train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

train_dataset = tf.data.Dataset.from_tensor_slices(train_images)\
    .shuffle(60000).batch(32)

latent_dim = 100  # latent space’s dimensionality
lr = 0.0002
# coefficients for the Adam optimizer
beta1 = 0.5
beta2 = 0.999
num_epochs = 10

# Loss function
adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Optimizers
optimizer_G = tf.keras.optimizers.Adam(lr=lr, beta_1=beta1, beta_2=beta2)
optimizer_D = tf.keras.optimizers.Adam(lr=lr, beta_1=beta1, beta_2=beta2)

generator = Generator()
discriminator = Discriminator()

# Training loop
for epoch in range(num_epochs):
    for i, real_images in enumerate(train_dataset):
        # Sample noise as generator input
        z = tf.random.normal([real_images.shape[0], latent_dim])
        # Generate a batch of images
        with tf.GradientTape() as tape:
            fake_images = generator(z, training=True)
            # Measure discriminator's ability to classify real and fake images
            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(fake_images, training=True)
            d_loss = adversarial_loss(tf.ones_like(real_output), real_output) + \
                     adversarial_loss(tf.zeros_like(fake_output), fake_output)
        # Backward pass and optimize discriminator
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        optimizer_D.apply_gradients(zip(grads, discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as tape:
            gen_images = generator(z, training=True)
            fake_output = discriminator(gen_images, training=True)
            g_loss = adversarial_loss(tf.ones_like(fake_output), fake_output)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        optimizer_G.apply_gradients(zip(grads, generator.trainable_variables))

        # Progress Monitoring
        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch {i+1}/{len(train_images)//32} "
                f"Discriminator Loss: {d_loss.numpy():.4f} Generator Loss: {g_loss.numpy():.4f}"
            )
    # Save generated images for every epoch
    if (epoch + 1) % 10 == 0:
        z = tf.random.normal([16, latent_dim])
        generated = generator(z, training=False)
        plt.imshow(np.transpose(generated.numpy(), (1, 2, 0)))
        plt.axis("off")
        plt.show()
