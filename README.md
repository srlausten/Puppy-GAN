# GAN Dog Image Generation

This project uses a Generative Adversarial Network (GAN) to generate images of dogs. The GAN consists of two neural networks: the Generator and the Discriminator. The Generator creates images, and the Discriminator tries to distinguish between real and generated images.

## Dataset

The Stanford Dogs Dataset is used for training. It contains images of various dog breeds.

## Model Architecture

### Generator
The Generator network takes random noise as input and generates realistic images. It consists of several layers of transposed convolutions, batch normalization, and ReLU activations.

### Discriminator
The Discriminator network takes an image as input and outputs a probability indicating whether the image is real or generated. It consists of several layers of convolutions, leaky ReLU activations, and dropout for regularization.

## Training

The models are trained using the Adam optimizer. The training process alternates between training the Discriminator and the Generator.

### Hyperparameters
- Batch size: 32
- Learning rate (Generator): 0.001
- Learning rate (Discriminator): 0.0005
- Epochs: 10
- Latent vector size (nz): 128

### Loss Values
The loss values for both the Generator and the Discriminator were recorded throughout the training process. Both losses showed a decreasing trend, indicating that the models were learning to generate and discriminate images more effectively over time.

## Results

Below are some examples of generated images after training:
(Needs longer training)
![alt text](images/image_735.png)

## Conclusion

The GAN has started learning the data distribution but requires further training to produce more realistic and sharp images. Increasing the number of training epochs and fine-tuning the network architecture could improve the quality of the generated images.

## Usage

To generate images, run the `generate_and_save_images` function provided in the code. This will create 10,000 images and save them in a zip file named `images.zip`.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

## Instructions

1. Clone the repository.
2. Ensure you have the required packages installed.
3. Place the Stanford Dogs Dataset in the appropriate directory.
4. Run the training script to train the GAN.
5. Use the provided function to generate and save images.

## License

This project is licensed under the MIT License.
