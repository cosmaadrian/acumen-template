from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch import nn
from .outputs import MultiHead, ModelOutput


class MnistCNN(torch.nn.Module):
    def __init__(self, args):
        super(MnistCNN, self).__init__()
        self.args = args
        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * 7 * 7, 2 * self.args.model_args.embedding_size)
        self.embedding = nn.Linear(
                2 * self.args.model_args.embedding_size, 
                self.args.model_args.embedding_size,
                bias = False
            )
        
        self.outputs = MultiHead(self.args)
        

    def forward(self, batch):
        x = self.conv1(batch['image'])
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        x = self.embedding(nn.functional.gelu(output))

        model_output = ModelOutput(representation = x)
        classification_output = self.outputs(model_output)

        return {'model_output': model_output, **classification_output}

