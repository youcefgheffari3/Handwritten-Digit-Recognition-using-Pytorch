# ✍️ Handwritten Digit Recognition with PyTorch

This project demonstrates how to build and train a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset with PyTorch. It includes data loading, model design, training, evaluation, and saving the trained model.

## 📁 Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is used in this project. It consists of 70,000 grayscale images of handwritten digits (0–9), each sized 28x28 pixels.

- **60,000** training images
- **10,000** test images

### Sample Digits

['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


## 🚀 Features

- Loading and normalizing the MNIST dataset
- CNN architecture using `torch.nn`
- Model training and evaluation on GPU (if available)
- Saving and reloading the trained model
- Achieves ~96% accuracy on the test set

## 🧪 Model Architecture

The CNN consists of:

- 2 Convolutional layers (`Conv2d`)
- Max pooling and dropout for regularization
- 1 Fully connected (hidden) layer
- 1 Output layer with 10 classes and softmax activation

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```


