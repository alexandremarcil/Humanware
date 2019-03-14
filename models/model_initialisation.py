from models.baselines import ConvModel, BaselineCNNdropout
from models.customhead import CustomHead
from models.resnet import resnet18, resnet50, resnet101, resnet152
from models.vgg import VGG


def initialize_model(model_name, num_dense_layers=2, dropout=0):
    '''
    Initialise a model with a custom head
    to predict both sequence length and digits

    Parameters
    ----------
    model_name : str
        Model Name can be either:
        ResNet
        VGG
        ConvNet
        BaselineCNN_dropout

    Returns
    -------
    model : object
        The model to be initialize

    '''

    if model_name[:3] == "VGG":
        model = VGG(model_name, num_classes=7)
        model.classifier = CustomHead(512)

    elif model_name[:6] == "ResNet":
        if model_name == "ResNet18":
            model = resnet18(num_classes=7)
            model.linear = CustomHead(512)

        elif model_name == "ResNet34":
            model = resnet18(num_classes=7)
            model.linear = CustomHead(512)

        elif model_name == "ResNet50":
            model = resnet50(num_classes=7)
            model.linear = CustomHead(512 * 4)

        elif model_name == "ResNet101":
            model = resnet101(num_classes=7)
            model.linear = CustomHead(512 * 4)

        elif model_name == "ResNet152":
            model = resnet152(num_classes=7)
            model.linear = CustomHead(512 * 4)

    elif model_name == "ConvNet":
        model = ConvModel(num_dense_layers=num_dense_layers, dropout=dropout)

    elif model_name == "BaselineCNNdropout":
        model = BaselineCNNdropout(num_classes=7, p=dropout)
        model.fc2 = CustomHead(4096)

    return model
