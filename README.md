# Satellite to Map Image Translation using GANs

This project implements a Generative Adversarial Network (GAN) for translating satellite imagery to map-style images. The model is based on the pix2pix architecture and is implemented using Keras.

![Example of image translation](https://github.com/BhoumikPatidar/Generative-Adversarial-Network-For-Image-Translation/blob/main/results.png)

## Project Overview

This project aims to automate the process of converting satellite imagery into map-style images using deep learning techniques. We utilize a conditional GAN architecture, specifically inspired by the pix2pix model, to learn the mapping between satellite images and their corresponding map representations.

The model is trained on paired datasets of satellite images and their corresponding map images. Once trained, it can generate map-like images from new satellite imagery inputs.

## Model Architecture

The GAN consists of two main components: a generator and a discriminator.

### Generator

The generator follows a U-Net architecture:
- Encoder: 7 downsampling blocks
- Bottleneck: 1 block
- Decoder: 7 upsampling blocks with skip connections

Each encoder block consists of:
- Convolution
- Batch Normalization
- LeakyReLU activation

Each decoder block consists of:
- Transposed Convolution
- Batch Normalization
- Dropout (in the first 3 blocks)
- ReLU activation

### Discriminator

The discriminator is a PatchGAN classifier, which aims to classify if each N×N patch in an image is real or fake.

It consists of 5 convolutional layers with the following structure for each layer:
- Convolution
- BatchNormalization (except the first layer)
- LeakyReLU activation

The final layer uses a sigmoid activation to produce the classification output.

## Training Process

The training process follows these steps:

1. Load and preprocess the dataset
2. Define and compile the discriminator
3. Define the generator
4. Define the combined GAN model
5. Train the model for a specified number of epochs:
   - Generate a batch of "fake" images using the generator
   - Train the discriminator on real and fake images
   - Train the generator via the combined GAN model

The model uses the following loss functions:
- Discriminator: Binary Cross-Entropy
- Generator: 
  - Adversarial loss: Binary Cross-Entropy
  - L1 loss: Mean Absolute Error between generated and target images

Training hyperparameters:
- Learning rate: 0.0002
- Beta1 for Adam optimizer: 0.5
- Batch size: 1 (as per the code, but this can be adjusted)
- Number of epochs: 100 (adjustable in the `train` function)

## Future Improvements

1. Experiment with different architectures, such as CycleGAN for unpaired image translation
2. Implement data augmentation techniques to improve generalization
3. Try different loss functions, such as perceptual loss, to improve the quality of generated images
4. Extend the model to handle higher resolution images
5. Explore the use of attention mechanisms to focus on important features during translation
