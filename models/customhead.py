import torch
import torch.nn as nn


class CustomHead(nn.Module):
    
    '''
    Create a nn.module to predict all digits and sequence lentgh at the same time
    
    '''
    
    def __init__(self, nb_input):
        
        '''

        Parameters
        ----------
        nb_input : int
            Number of input features should change depending of the model used for the body
        '''
        
        super(CustomHead, self).__init__() 
        self.digits_lenght = nn.Linear(nb_input, 7)
        self.digit_layer1 = nn.Linear(nb_input, 10)
        self.digit_layer2 = nn.Linear(nb_input, 10)
        self.digit_layer3 = nn.Linear(nb_input, 10)
        self.digit_layer4 = nn.Linear(nb_input, 10)
        self.digit_layer5 = nn.Linear(nb_input, 10)
    
    def forward(self, x):
        
        digits = []
        digits_lenght = self.digits_lenght(x)
        for layer in [self.digit_layer1, self.digit_layer2, self.digit_layer3, self.digit_layer4, self.digit_layer5]:
            digits.append(layer(x))
            
        return digits_lenght, digits