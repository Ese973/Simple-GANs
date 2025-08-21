# Simple-GANs
Simple pair of one-layer GANs used to generate simple 2x2 black and white images.

Code was repoduced from https://github.com/luisguiserrano/gans. Used for reinforcing concepts of GANs (Generative Adversial Networks) without the use of machine learning packages.

Background for understanding and using the model:
1. In "Slanted Land" faces are slanted at a 45 degree angle. This is represented by a 2x2 image with darker top left and bottom right corners (diagonal, on a scale of 0-1 with closer to 1 being darker).

<div align = "center">
<img width="790" height="186" alt="Image" src="https://github.com/user-attachments/assets/421e1ee1-c605-494a-9626-18b66eed4a60" />
</div>

2. After building the discriminator to identify real and fake faces from slanted land, the generator will try to feed fake faces to the discriminator using a random input vector (z, scale 0-1).
   
3. The discriminator will output a probability of the image being fake. It does this by summing the values of each vector (4 in total) + the bias value and putting it in a sigmoid function. If discrminator was perfect it would want to output a 0. Therefore are loss function needed is error = -log(1-probability). For the generator it is error = -log(probability) as it would ideally want to be a 1. Both errors are used to adjust the weights of each network by using the derivatives.

4. After a certain amount of iterations, the generator will be able to generate real enough faces (slanted) that the discrimintor will not be able to detect it. That is when the model has been trained!

