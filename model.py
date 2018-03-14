import os
import shutil
import torch
import torch.nn as nn
from torch import optim
from utils import to_var
from neural_utils import ContrastiveLoss, PyTorchSiameseNetwork


class SiameseModel(object):

    def __init__(self,
                 input_image_shape,
                 output_size=2,
                 margin=0.2,
                 learning_rate=0.01,
                 weight_decay=0.0001,
                 momentum=0.9):
        
        self.target_shape = input_image_shape
        self.embedding_size = output_size
        
        # Load the model and nn modules.
        if torch.cuda.is_available():
            self.network = PyTorchSiameseNetwork(output_size)
            self.network = self.network.cuda()
            
            self.criterion = ContrastiveLoss(margin=margin).cuda()
            self.optimiser = optim.SGD(self.network.parameters(), 
                                       lr=learning_rate,
                                       weight_decay=weight_decay,
                                       momentum=momentum)
        else:
            raise ValueError('You need a GPU for this network!')

    def inference(self, x):
        """Here the net computes an embedding for the input image(s)."""
        
        self.network.eval()
        
        # Convert to variables on the GPU.
        width, height, channels = self.target_shape
        pytorch_shape = (-1, channels, width, height)
        
        # NumPy to PyTorch.
        x = to_var(torch.from_numpy(x.reshape(pytorch_shape)))
        
        # Get embedding.
        embedding = self.network.forward_once(x)
        
        # Convert result to NumPy and return.
        return embedding.cpu().data.numpy()

    def optimise_batch(self, batch_left, batch_right, batch_similar):
        
        self.network.train()
        
        # Initialise the gradients buffers.
        self.optimiser.zero_grad()

        # Convert to variables on the GPU.
        width, height, channels = self.target_shape
        pytorch_shape = (-1, channels, height, width)
        
        # Reshape for pytorch model.
        batch_left = batch_left.reshape(pytorch_shape)
        batch_right = batch_right.reshape(pytorch_shape)
        
        # NumPy to PyTorch.
        batch_left = to_var(torch.from_numpy(batch_left))
        batch_right = to_var(torch.from_numpy(batch_right))
        batch_similar = to_var(torch.from_numpy(batch_similar))  
        
        # Get encoding vectors.
        encoding_left, encoding_right = self.network(batch_left, batch_right)
        
        # Loss calculated and backpropagate error.
        loss = self.criterion(encoding_left, encoding_right, batch_similar)
        loss.backward()
        self.optimiser.step()
        
        return loss.data[0]

    def save(self, path, is_best=False):
        state = {
            'network': self.network.state_dict(),
            'optimiser': self.optimiser.state_dict()
        }
        state_path = os.path.join(path, 'model.pth.tar')
        torch.save(state, state_path)
        
        if is_best:
            best_path = os.path.join(path, 'model_best.pth.tar')
            shutil.copyfile(state_path, best_path)
        
    def load(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            self.network.load_state_dict(checkpoint['network'])
            self.optimiser.load_state_dict(checkpoint['optimiser'])
            print("=> loaded checkpoint '{}'".format(path))
        else:
            print("=> no checkpoint found at '{}'".format(path))