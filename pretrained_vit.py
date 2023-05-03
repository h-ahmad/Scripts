import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32 
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
    
class FCLayer(nn.Module):
    def __init__(self, model, model_type, num_classes):
        super(FCLayer, self).__init__()
        self.model = model    
        if model_type == 'vit_b_16':
            num_ftrs = 768
        elif model_type == 'vit_b_32':
            num_ftrs = 768
        elif model_type == 'vit_l_16':
            num_ftrs = 1024
        elif model_type == 'vit_l_32':
            num_ftrs = 1024
        else:
            print('Invilade model type!!!')
        
        self.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        x = self.model(x)['getitem_5']
        out = self.fc(x)
        return out

class PTVIT(nn.Module):
    def __init__(self, model_type, num_cls=15):
        super(PTVIT, self).__init__()
        if model_type == 'vit_b_16':
            self.model = vit_b_16(pretrained=True)
        elif model_type == 'vit_b_32':
            self.model = vit_b_32(pretrained=True)
        elif model_type == 'vit_l_16':
            self.model = vit_l_16(pretrained=True)
        elif model_type == 'vit_l_32':
            self.model = vit_l_32(pretrained=True)
        else:
            print('Invilade model type!!!')
        
        # freeze all the network except the final layer
        # for param in self.model.parameters():
        #  	param.requires_grad = False
        
        # These two lines not needed but, you would use them to work out which node you want
        nodes, eval_nodes = get_graph_node_names(self.model)
        #print('model nodes:', nodes)
        self.features_encoder = create_feature_extractor(self.model, return_nodes=['encoder'])
        #print('features_encoder:', self.features_encoder)
        self.features_vit_flatten = create_feature_extractor(self.model, return_nodes=['getitem_5'])
        self.features_fc = create_feature_extractor(self.model, return_nodes=['heads'])
        self.model_final = FCLayer(self.features_vit_flatten, model_type, num_cls)

    def forward(self, input):
        #out1 = self.features_vit_flatten(input)
        #print('out1:', out1['getitem_5'].shape)
        out = self.model_final(input)
        return out

if __name__ == "__main__":    
    input = torch.randn(2,3,224,224).type(torch.FloatTensor)
    print('input:', input.shape)
    
    model_type = 'vit_b_16'
    
    model = PTVIT(model_type)
    
    #print(model)
    
    #num_parameters = sum([np.prod(list(p.size())) for p in model.parameters()])
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print('Number of model params: %f [M]' % num_parameters)
    
    output = model(input)
    print('output:', output.shape)
