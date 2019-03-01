from models.baselines import BaselineCNN, ConvNet, BaselineCNN_dropout
from models.customhead import CustomHead
from models.resnet import ResNet18, ResNet50, ResNet101, ResNet152
from models.vgg import VGG


def initialize_model(model_name):
    '''
    Initialise a model with a custom head to predict both sequence length and digits

    Parameters
    ----------
    model_name : str
        Model Name can be either:
        ResNet
        VGG
        BaselineCNN
        ConvNet
        BaselineCNN_dropout
        
    Returns
    -------
    model : object
        The model to be initialize 

    '''

        
    if model_name[:3] == "VGG":
        model = VGG(model_name, num_classes = 7)
        model.classifier = CustomHead(512)
        
    elif  model_name[:6] == "ResNet":
        if model_name == "ResNet18":
            model = ResNet18(num_classes = 7)
            model.linear = CustomHead(512)

        elif model_name == "ResNet34":
            model = ResNet18(num_classes = 7)
            model.linear = CustomHead(512)

        elif model_name == "ResNet50":
            model = ResNet50(num_classes = 7)
            model.linear = CustomHead(512*4)

        elif model_name == "ResNet101":
            model = ResNet101(num_classes = 7)
            model.linear = CustomHead(512*4)

        elif model_name == "ResNet152":
            model = ResNet152(num_classes = 7)
            model.linear = CustomHead(512*4)

    elif model_name == "ConvNet":
        model =  ConvNet(num_classes = 7)
        model.fc = CustomHead(4608)
        
    elif model_name == "BaselineCNN":
        model =  BaselineCNN(num_classes = 7)
        model.fc2 = CustomHead(4096)
        
    elif model_name == "BaselineCNN_dropout":
        model =  BaselineCNN_dropout(num_classes = 7, p = 0.5)
        model.fc2 = CustomHead(4096)
        
    return model