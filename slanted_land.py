# Import packages
import numpy as np
from numpy import random
from matplotlib import pyplot as plt


# Create a function to draw my images
def observe_samples(samples, m, n):
    fig, axes = plt.subplots(
        figsize=(10, 10), nrows=m, ncols=n, sharey=True, sharex=True
    )
    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(1 - img.reshape((2, 2)), cmap="Greys_r")
    return fig, axes


# Examples of faces
faces = [
    np.array([1, 0, 0, 1]),
    np.array([0.9, 0.1, 0.2, 0.8]),
    np.array([0.9, 0.2, 0.1, 0.8]),
    np.array([0.8, 0.1, 0.2, 0.9]),
    np.array([0.8, 0.2, 0.1, 0.9]),
]

_ = observe_samples(faces, 1, 4)

# Now we create noisy images or fake faces
noise = [np.random.randn(2, 2) for i in range(20)]

_ = observe_samples(noise, 4, 5)


# Create a function to return a random image for later
def generate_random_image():
    return [
        np.random.random(),
        np.random.random(),
        np.random.random(),
        np.random.random(),
    ]


# Now we are going to build the neural network
# Activate sigmoid function for predictions
def sigmoid(x):
    return np.exp(x) / (1.0 + np.exp(x))


# Let's build the discriminator
class Discriminator:
    def __init__(self):
        self.weights = np.array([np.random.normal() for i in range(4)])
        self.bias = np.random.normal()

    # Forward pass
    def forward(self, x):
        return sigmoid(np.dot(x, self.weights) + self.bias)

    # We generate the error from the prediciton based off the image
    def error_from_image(self, Image):
        prediction = self.forward(Image)
        # Since the prediction we want is 1, the error is -log(prediction)
        return -np.log(prediction)

    # We need to get the derivatives from the error so we can adust the model
    def derivatives_from_image(self, image):
        prediction = self.forward(image)
        derivatives_weights = -image * (1 - prediction)
        derivative_bias = -(1 - prediction)
        return derivatives_weights, derivative_bias

    # Then we update the model
    def update_from_image(self, x):
        ders = self.derivatives_from_image(x)
        self.weights -= learning_rate * ders[0]
        self.bias -= learning_rate * ders[1]

    # Error from fake images/faces
    def error_from_noise(self, noise):
        prediction = self.forward(noise)
        # We want the prediction to be 0, so the error is -log(1-prediction)
        return -np.log(1 - prediction)

    # Get derivatives from fake images (noise)
    def derivatives_from_noise(self, noise):
        prediction = self.forward(noise)
        derivatives_weights = noise * prediction
        derivative_bias = prediction
        return derivatives_weights, derivative_bias

    # Update the model based on noise
    def update_from_noise(self, noise):
        ders = self.derivatives_from_noise(noise)
        self.weights -= learning_rate * ders[0]
        self.bias -= learning_rate * ders[1]


# Now we build the generator
class Generator:
    def __init__(self):
        self.weights = np.array([np.random.normal() for i in range(4)])
        self.biases = np.array([np.random.normal() for i in range(4)])

    # Make the forward pass
    def forward(self, z):
        return sigmoid(z * self.weights + self.biases)

    # Obtain the error from our prediction
    def error(self, z, discriminator):
        x = self.forward(z)
        # We want the prediction to be 0, so the error is -log(1-prediction)
        y = discriminator.forward(x)
        return -np.log(y)

    # Obtain derivatives for the generator error
    def derivatives(self, z, discriminator):
        discriminator_weights = discriminator.weights
        discriminator_bias = discriminator.bias
        x = self.forward(z)
        y = discriminator.forward(x)
        factor = -(1 - y) * discriminator_weights * x * (1 - x)
        derivatives_weights = factor * z
        derivative_bias = factor
        return derivatives_weights, derivative_bias

    # Update the generator
    def update(self, z, discriminator):
        error_before = self.error(z, discriminator)
        ders = self.derivatives(z, discriminator)
        self.weights -= learning_rate * ders[0]
        self.biases -= learning_rate * ders[1]
        error_after = self.error(z, discriminator)


# Time to train the model
# Set random seed
np.random.seed(42)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# The GAN
D = Discriminator()
G = Generator()

# For the error plot
errors_discriminator = []
errors_generator = []

for epocn in range(epochs):
    for face in faces:
        # Update the discriminator weights from the real images/faces
        D.update_from_image(face)
        # Pick a random number to generate a fake image/face
        z = random.rand()
        # Calculate the discriminator error
        errors_discriminator.append(
            sum(D.error_from_image(face) + D.error_from_noise(z))
        )
        # Caculate the generator error
        errors_generator.append(G.error(z, D))
        # Build a fake image/face
        noise = G.forward(z)
        # Update the discriminator weights from the fake face
        D.update_from_noise(noise)
        # Update the generator weights from the fake face
        G.update(z, D)

# Generate error plots
plt.plot(errors_generator)
plt.title("Generator error function")
plt.legend("gen")
plt.show()
plt.plot(errors_discriminator)
plt.legend("disc")
plt.title("Discriminator error function")

# Generating Real Images
generated_images = []
for i in range(4):
    z = random.random()
    generated_image = G.forward(z)
    generated_images.append(generated_image)

_ = observe_samples(generated_images, 1, 4)

for i in generated_images:
    print(i)

# Observing the weights and biases of the generator and discriminator
print("Generator weights", G.weights)
print("Generator biases", G.biases)
print("Discriminator weights", D.weights)
print("Discriminator bias", D.bias)
