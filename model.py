import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitClassifierCNN(nn.Module):

    def __init__(self):
        super(DigitClassifierCNN, self).__init__()
        # Initiazlize the parent PyTorch class. Three convolutional layers.
        # The first one has 1 inchannel (only grayscale)
        # and 32 outchannels (32 different 3x3 filters to detect patterns).
        # The second layer takes those 32 feature maps and combines them
        # into 64 distinct patters like horizontal edges or vertical strokes.
        # Finally, the image is being downsampled by taking the maximum value
        # from every 2x2 block to reduce compute time.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Randomly turn off 25% of neurons on each step to prevent overfitting
        # during the training (stops from memorizing images)
        self.dropout = nn.Dropout(0.25)
        # Connected layer that takes the 1600 flattened spatial features and then
        # compresses them into 128 abstract concepts
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        # Produces output scores (logits), 1 for each digit
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First and second convolution block: Conv -> ReLU -> Pool
        # ReLu = Rectified Linear Unit
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten into 1D feature vector
        x = torch.flatten(x, 1)
        # Dense hidden layer with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output logits
        return self.fc2(x)
