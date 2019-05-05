import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from copy import deepcopy

class CLoss(torch.nn.Module):
    
    def __init__(self, org_content_feature_map, w):
        """
        Module that will be used to calculate the content loss for specific 
        convolutional layers in the network.        
        
        This is an implementation of equation one in the Leon paper. Pytorch autograd handles the backward
        pass in equation 2 for us.
        
        :param original: Feature map of previous layer for the original content image.
        :param w: weight for this layer in total loss.
        """
        super(CLoss, self).__init__()
        self.org_content_feature_map = org_content_feature_map
        self.w = w
    
    def forward(self, x):
        """
        This method calculates the mean squared error between the feature map of the content image
        and the feature map of the optimization image.
        
        :param x: Feature map from following convolutiona layer of image we are optimizing.
        """
        B, N, W, H = x.shape
        self.loss = torch.div(torch.sum(torch.pow(x - self.org_content_feature_map, 2)), B*N*W*H)
        
        # Propogate the original feature map through the network.
        return x
    
class SLoss(torch.nn.Module):
    
    def __init__(self, org_style_feature_map, w):
        """
        Module that will be used to calculate the style loss for specific convolutional layers in the network. This involves the 
        mean squared error between the gram matrix of the input image and the gram matrix of the style image.
        
        :param org_style: Feature map of previous layer for the origional style image.
        :param w: weight for this layer in total loss.
        """
        super(SLoss, self).__init__()
        self.org_style_feature_map = self.gram(org_style_feature_map)
        self.w = w
    
    def forward(self, x):
        """
        This method calculates the mean squared error between the gram matrix of the feature map
        for the style image and the feature map for the optimization image
        """
        B, N, W, H = x.shape
        self.loss = torch.div(torch.sum(torch.pow(self.gram(x) - self.org_style_feature_map, 2)), 4 * ((B*N)**2) * ((W*H)**2))
        
        # Propogate the original feature map through the network.
        return x
        
    @staticmethod
    def gram(m):
        """
        This method calculates the gram matrix for a given feature map.
        
        Given a feature map F that is (N, M) where N is the number of features and M is the width*height of the filters the
        gram matrix is calculated by by F@F^T. This should result in a Gram matrix of size (N, N). Most feature maps passed 
        through a network are in the shape (Batch, Num Filter, Width Height), so transformations to the desired shape will 
        need to occur.
        
        F = [x1, 
             x2,
             x3, 
             x4] is (N, M)
        F^t = [x1, x2, x3, x4] is (M, N)
        F@F^t is (N,N) = [x1x1 x1x2 x1x3 x1x4,
                          ...................,
                          ...................,
                          x4x1 x4x2 x4x3 x4x4]
        
        :param m: Is a feature map of (Batch, Number of Filter, Width, Height) of optimization image through network.
        """
        B, N, W, H = m.shape
        N, M = B*N, W*H
        feature_map_cols = m.reshape(N, M)
        
        # Calculating F@F^T
        return feature_map_cols@feature_map_cols.t()
    
    
class StyleTransferLoss(torch.nn.Module):
    
    def __init__(self, style_losses, alpha, content_losses, beta):
        """
        This module represents the total loss function for the style transfer network.
        """
        super(StyleTransferLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.style_losses = style_losses
        self.content_losses = content_losses        
        
    def forward(self):
        """
        This method calculates the loss throughout the whole network.
        """
        self.style_component = np.sum([s.loss*s.w for s in self.style_losses])
        self.content_component = np.sum([c.loss*c.w for c in self.content_losses])
        return self.beta*self.style_component + self.alpha*self.content_component
    
    
    
class StyleTransfer(torch.nn.Module):
    
    def __init__(self, org_mod, content_image, style_image, content_ixs, style_ixs, style_weights=None,
                 alpha=1, beta=1):
        """
        This module represents a style transfer network. It takes in a pre existing pretrained 
        network and runs an image through it keeping track of pertinant feature maps relating to the
        content and style of the image.
        
        :param org_mod: Convolutional model with pretrained weights.
        :param content_ixs: List of integers representing the index of the convolutional layer to capture the content
        loss information from.
        :param style_ixs: List of integers representing the index of the convolutional layer to capture the style
        loss information from.
        """
        super(StyleTransfer, self).__init__()
        
        self.content_ixs = content_ixs
        self.style_ixs = style_ixs
        self.content_losses = []
        self.style_losses = []
        self.alpha = alpha
        self.beta = beta
        self.style_weights = style_weights
        
        assert(len(self.style_weights) == len(self.style_ixs))
        
        # We need to recreate the model but with the new content losses and style losses.
        conv_ix = 0
        self.model = torch.nn.Sequential(
        )
        for ix, layer in enumerate(deepcopy(org_mod)):
            
            
            # If the last layer added was a convolutional layer then we need to check if we are adding content or style loss to it.
            if isinstance(layer, torch.nn.Conv2d):
                self.model.add_module(str(ix), layer)
                # If we are at a convolutional index in which we want to track add the content or style loss to it.
                if conv_ix in self.content_ixs:
                    print("Found content! ", conv_ix)
                    # Calculate feature map for content image at current model.
                    content_feature_map = org_mod[:ix+1](content_image).detach()
                    closs = CLoss(content_feature_map, w=1)
                    self.content_losses.append(closs)
                    self.model.add_module('content_loss_{}'.format(conv_ix), closs)
                if conv_ix in self.style_ixs:
                    print("Found style! ", conv_ix, ' weight: {}'.format(self.style_weights[self.style_ixs.index(conv_ix)]))
                    # Calculate feature map for style image at current model.
                    loss_feature_map = org_mod[:ix+1](style_image).detach()
                    sloss = SLoss(loss_feature_map, w=self.style_weights[self.style_ixs.index(conv_ix)])
                    self.style_losses.append(sloss)
                    self.model.add_module('style_loss_{}'.format(conv_ix), sloss)
                
                # See if we are done generating the new model.
                if conv_ix >= max(self.style_ixs) and conv_ix >= max(self.content_ixs):
                    break
                else:
                    conv_ix+=1  
            # PYTORCH WAS THORWING ERRORS IF I HAD IN PLACE FOR THE RELUS! Not sure why???
            elif isinstance(layer, torch.nn.ReLU):
                self.model.add_module(str(ix), torch.nn.ReLU(inplace=False))
            else:
                self.model.add_module(str(ix), layer)
                

    
    def forward(self, x):
        x = self.model(x)
        return x