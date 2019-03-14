import torch.nn as nn
import torch.nn.functional as F
from models.customhead import CustomHead


class ConvModel(nn.Module):

    def __init__(self, num_dense_layers, dropout=0.2):
        super(ConvModel, self).__init__()

        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=48,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48,
                      out_channels=64,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(dropout)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=160,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(dropout)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160,
                      out_channels=192,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192,
                      out_channels=192,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(dropout)
        )

        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192,
                      out_channels=192,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192,
                      out_channels=192,
                      kernel_size=5,
                      padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(dropout)
        )

        hidden9 = nn.Sequential(
            nn.Linear(192 * 7 * 7, 3072),
            nn.ReLU()
        )
        hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )

        self._features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )

        if num_dense_layers == 0:
            self._classifier = nn.Sequential()
            input_features = 192 * 7 * 7

        elif num_dense_layers == 1:
            self._classifier = nn.Sequential(hidden9)
            input_features = 3072

        elif num_dense_layers == 2:
            self._classifier = nn.Sequential(hidden9, hidden10)
            input_features = 3072

        self.custom_output = CustomHead(input_features)

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), 192 * 7 * 7)
        x = self._classifier(x)
        x = self.custom_output(x)
        return x


class BaselineCNN(nn.Module):  # Achieves ~91%

    def __init__(self, num_classes):
        '''
        Placeholder CNN
        '''
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7744, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        '''
        Forward path.

        Parameters
        ----------
        x : ndarray
            Input to the network.

        Returns
        -------
        x : ndarray
            Output to the network.

        '''

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten based on batch size
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class BaselineCNNdropout(nn.Module):

    def __init__(self, num_classes, p=0.5):
        '''
        Placeholder CNN
        '''
        super(BaselineCNNdropout, self).__init__()

        self.p = p
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(self.p)

        self.fc1 = nn.Linear(7744, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        '''
        Forward path.

        Parameters
        ----------
        x : ndarray
            Input to the network.

        Returns
        -------
        x : ndarray
            Output to the network.

        '''

        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        # Flatten based on batch size
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
