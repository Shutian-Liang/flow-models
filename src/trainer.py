# instead of using pytorch lightening, the common loop
import os
import math
import torch
import wandb
from torch import optim
from tqdm import tqdm
from dataset import CIFAR10Dataset
from utils import checkandcreate

class Trainer:
    def __init__(self, generative, params: dict):
        """the trainer for later 

        Args:
            generative (_type_): flow or diffusion 
            params (_type_): the basic parameters for training, such as learning rate, batch size, etc.
        """
        self.generative = generative
        self.params = params
        self.optimizer = optim.AdamW(self.model.parameters(), lr=params['learning_rate'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generative.to(self.device)
        self.dataset = CIFAR10Dataset(params['data_path'], params['batch_size'])
        self.dataloader = self.dataset.train_dataloader()
        
        # optimizers
        self.optimizer = optim.AdamW(self.generative.parameters(), lr=params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=params['epochs'], power=2.0)
        self.epochs = params['epochs']
        
    def train(self):
        self.generative.train()
        for epoch in range(self.epochs):
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                loss = self.generative(images, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if batch % self.params['log_interval'] == 0:
                    wandb.log({"loss": loss.item(), "epoch": epoch + 1, "batch": batch})
                    print(f"Epoch [{epoch+1}/{self.epochs}], Batch [{batch}], Loss: {loss.item():.4f}")

            self.scheduler.step()
            self.evaluate(epoch)
            self.save_checkpoints(epoch)

    def evaluate(self, epoch):
        self.generative.eval()
        labels = torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]).to(self.device)
        with torch.no_grad():
            sample = torch.randn((self.params['data_params']['batch_size'], 3, 
                                  self.params['data_params']['img_size'], 
                                  self.params['data_params']['img_size'])).to(self.device)
            output = self.generative.model.decode(sample, labels).clamp(0,1)
        path = os.path.join(self.params['output_dir'], f"/samples/epoch_{epoch}.png")
        checkandcreate(os.path.dirname(path))
        self.dataset.show_images(output, epoch, path)
    
    def save_checkpoints(self, epoch):
        if (epoch+1) % self.params['save_interval'] == 0:
            checkpoint_path = os.path.join(self.params['output_dir'], f"checkpoints/epoch_{epoch}.pth")
            checkandcreate(os.path.dirname(checkpoint_path))
            torch.save({
                'epoch': epoch, 
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        else:
            pass