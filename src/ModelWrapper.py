# https://github.com/oscarhoffmann3487/Flow_Matching_vs_DDPM_on_CIFAR-10/blob/main/helpers/ModelWrapper.py
from zuko.utils import odeint
import torch
import torch.nn as nn

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x, t, y = None):
      return self.model(x, t, y)

    def wrapper(self, t, x):
      #adjust t for odesolving to model.
      t = t * torch.ones(len(x), device=x.device)
      return self(x, t,  self.labels)

    def decode(self, x_0, labels = None):
      if labels is not None:
        self.labels = labels
      #solve ode to get datapoint. integrate from 0 to 1 to go from noise to data
      return odeint(self.wrapper, x_0, 0, 1, self.parameters())