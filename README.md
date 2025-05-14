# CIFAR-10 Image Classification with ResNet
MSCS-335 Final Project

Goal: Create a classifier network for the CIFAR-10 dataset

We utilize the ResNet architecture proposed by He, Kaiming et al. in their "Deep residual learning for image recognition" paper. Using both the ResNet-34 model & the ResNet-50 model, we achieved a stable accuracy of around 82%, with a peak accuracy of 85%.

The model was trained on a batch size of 32 RGB images using Stochastic Gradient Descent as the optimizer, and Cross Entropy loss. Most models were trained for 40 epochs, although they stopped seeing improvement after 20.

To improve acuraccy we attempted preprocessing the images with data augmentation - by applying random transformations to each image we hoped to help the model understand the relationship between features better.

### Data Augmentation
<img src="https://github.com/user-attachments/assets/293da8ce-9a26-4c3d-91a2-94424d2e153d" width="400"></img>\
<img src="https://github.com/user-attachments/assets/83c3d8ec-d1cd-4927-a98f-3b2efcf7deb2" width="300"></img>

### ResNet-34 Results:
<img src="https://github.com/user-attachments/assets/0f4b3977-9c67-4dc1-abfe-61aff4767de7" width="400"></img>
### ResNet-50 Results:
<img src="https://github.com/user-attachments/assets/ec3d1fdf-d202-4d03-adde-00501138eb31" width="400"></img>

## References: 
>Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images.(2009)." Sep. 2009.\
>\
>He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.\
>\
>Ahmed, Nouman. “Writing Resnet from Scratch in Pytorch.” DigitalOcean, DigitalOcean, 16 Sept. 2024, www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch.
>\
>\
>Dosovitskiy, Alexey, et al. ‘Discriminative Unsupervised Feature Learning with Convolutional Neural Networks’. Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1, MIT Press, 2014, pp. 766–774. NIPS’14.



