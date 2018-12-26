import os
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import *
from tqdm import *
from dataset import *

__all__ = ['loss_fn', 'Trainer']



def loss_fn(original_seq,recon_seq,f_mean,f_logvar,z_post_mean,z_post_logvar, z_prior_mean, z_prior_logvar):
    batch_size = original_seq.size(0)
    mse = F.mse_loss(recon_seq,original_seq,reduction='sum');
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)
    return (mse + kld_f + kld_z)/batch_size, kld_f/batch_size, kld_z/batch_size


class Trainer(object):
    def __init__(self,model,train,test,trainloader,testloader, test_f_expand,
                 epochs=100,batch_size=64,learning_rate=0.001,nsamples=1,sample_path='./sample',
                 recon_path='./recon/', transfer_path = './transfer/', 
                 checkpoints='model.pth', style1='image1.sprite', style2='image2.sprite', device=torch.device('cuda:0')):
        self.trainloader = trainloader
        self.train = train
        self.test = test
        self.testloader = testloader
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.optimizer = optim.Adam(self.model.parameters(),self.learning_rate)
        self.samples = nsamples
        self.sample_path = sample_path
        self.recon_path = recon_path
        self.transfer_path = transfer_path
        self.test_f_expand = test_f_expand
        self.epoch_losses = []

        self.image1 = torch.load(self.transfer_path + 'image1.sprite')['sprite']
        self.image2 = torch.load(self.transfer_path + 'image2.sprite')['sprite']
        self.image1 = self.image1.to(device)
        self.image2 = self.image2.to(device)
        self.image1 = torch.unsqueeze(self.image1,0)
        self.image2= torch.unsqueeze(self.image2,0)
    
    def save_checkpoint(self,epoch):
        torch.save({
            'epoch' : epoch+1,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'losses' : self.epoch_losses},
            self.checkpoints)
        
    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def sample_frames(self,epoch):
        with torch.no_grad():
            _,_,test_z = self.model.sample_z(1, random_sampling=False)
            print(test_z.shape)
            print(self.test_f_expand.shape)
            test_zf = torch.cat((test_z, self.test_f_expand), dim=2)
            recon_x = self.model.decode_frames(test_zf) 
            recon_x = recon_x.view(self.samples*8,3,64,64)
            torchvision.utils.save_image(recon_x,'%s/epoch%d.png' % (self.sample_path,epoch))
    
    def recon_frame(self,epoch,original):
        with torch.no_grad():
            _,_,_,_,_,_,_,_,recon = self.model(original) 
            image = torch.cat((original,recon),dim=0)
            image = image.view(2*8,3,64,64)
            os.makedirs(os.path.dirname('%s/epoch%d.png' % (self.recon_path,epoch)),exist_ok=True)
            torchvision.utils.save_image(image,'%s/epoch%d.png' % (self.recon_path,epoch))

    def style_transfer(self,epoch):
        with torch.no_grad():
            conv1 = self.model.encode_frames(self.image1)
            conv2 = self.model.encode_frames(self.image2)
            _,_,image1_f = self.model.encode_f(conv1)
            image1_f_expand = image1_f.unsqueeze(1).expand(-1,self.model.frames,self.model.f_dim)
            _,_,image1_z = self.model.encode_z(conv1,image1_f)
            _,_,image2_f = self.model.encode_f(conv2)
            image2_f_expand = image2_f.unsqueeze(1).expand(-1,self.model.frames,self.model.f_dim)
            _,_,image2_z = self.model.encode_z(conv2,image2_f)
            image1swap_zf = torch.cat((image2_z,image1_f_expand),dim=2)
            image1_body_image2_motion = self.model.decode_frames(image1swap_zf)
            image1_body_image2_motion = torch.squeeze(image1_body_image2_motion,0)
            image2swap_zf = torch.cat((image1_z,image2_f_expand),dim=2)
            image2_body_image1_motion = self.model.decode_frames(image2swap_zf)
            image2_body_image1_motion = torch.squeeze(image2_body_image1_motion,0)
            image1 = torch.squeeze(self.image1, 0)
            image2 = torch.squeeze(self.image2, 0)
            os.makedirs(os.path.dirname('%s/epoch%d/image1_body_image2_motion.png' % (self.transfer_path,epoch)),exist_ok=True)
            torchvision.utils.save_image(image1,'%s/epoch%d/image1.png' % (self.transfer_path,epoch))
            torchvision.utils.save_image(image2,'%s/epoch%d/image2.png' % (self.transfer_path,epoch))
            torchvision.utils.save_image(image1_body_image2_motion,'%s/epoch%d/image1_body_image2_motion.png' % (self.transfer_path,epoch))
            torchvision.utils.save_image(image2_body_image1_motion,'%s/epoch%d/image2_body_image1_motion.png' % (self.transfer_path,epoch))

    def train_model(self):
       self.model.train()
       for epoch in range(self.start_epoch,self.epochs):
           losses = []
           kld_fs = []
           kld_zs = []
           print("Running Epoch : {}".format(epoch+1))
           for i,dataitem in tqdm(enumerate(self.trainloader,1)):
               _,_,_,_,_,_,data = dataitem
               data = data.to(self.device)
               self.optimizer.zero_grad()
               f_mean, f_logvar, f, z_post_mean, z_post_logvar, z, z_prior_mean, z_prior_logvar, recon_x = self.model(data)
               loss, kld_f, kld_z = loss_fn(data, recon_x, f_mean, f_logvar, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar)
               loss.backward()
               self.optimizer.step()
               losses.append(loss.item())
               kld_fs.append(kld_f.item())
               kld_zs.append(kld_z.item())
           meanloss = np.mean(losses)
           meanf = np.mean(kld_fs)
           meanz = np.mean(kld_zs)
           self.epoch_losses.append(meanloss)
           print("Epoch {} : Average Loss: {} KL of f : {} KL of z : {}".format(epoch+1,meanloss, meanf, meanz))
           self.save_checkpoint(epoch)
           self.model.eval()
           self.sample_frames(epoch+1)
           _,_,_,_,_,_,sample = self.test[int(torch.randint(0,len(self.test),(1,)).item())]
           sample = torch.unsqueeze(sample,0)
           sample = sample.to(self.device)
           self.sample_frames(epoch+1)
           self.recon_frame(epoch+1,sample)
           self.style_transfer(epoch+1)
           self.model.train()
       print("Training is complete")

sprite = Sprites('./dataset/lpc-dataset/train', 6767)
sprite_test = Sprites('./dataset/lpc-dataset/test', 791)
batch_size = 25
loader = torch.utils.data.DataLoader(sprite, batch_size=batch_size, shuffle=True, num_workers=4)
device = torch.device('cuda:1')
vae = DisentangledVAE(f_dim=256, z_dim=32, step=256, factorised=True,device=device)
test_f = torch.rand(1,256, device=device)
test_f = test_f.unsqueeze(1).expand(1, 8, 256)
trainer = Trainer(vae, sprite, sprite_test, loader ,None, test_f, batch_size=30, epochs=500, learning_rate=0.0002, device=device)
trainer.load_checkpoint()
trainer.train_model()
