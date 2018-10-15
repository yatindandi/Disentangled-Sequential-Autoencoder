import os
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

__all__ = ['Sprites', 'loss_fn', 'Trainer']
class Sprites(torch.utils.data.Dataset):
    def __init__(self,path,size):
        self.path = path
        self.length = size;

    def __len__(self):
        return self.length
        
    def __getitem__(self,idx):
        return torch.load(self.path+'/%d.sprite' % (idx+1))


def loss_fn(original_seq,recon_seq,f_mean,f_logvar,z_mean,z_logvar):
    mse = F.mse_loss(recon_seq,original_seq,reduction='sum');
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))
    kld_z = -0.5 * torch.sum(1 + z_logvar - torch.pow(z_mean,2) - torch.exp(z_logvar))
    return mse + kld_f + kld_z
  

class Trainer(object):
    def __init__(self,model,train,test,trainloader,testloader,
                 epochs=50,batch_size=64,learning_rate=0.001,nsamples=8,sample_path='./sample',
                 recon_path='./recon', transfer_path = './transfer', 
                 checkpoints='model.pth', style1='image1.sprite', style2='image2.sprite'):
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
        self.test_f = torch.randn(self.samples,self.model.f_dim,device=self.device)
        self.test_z = torch.randn(self.samples,model.frames,model.z_dim,device=self.device)
        f_expand = self.test_f.unsqueeze(1).expand(-1,model.frames,model.f_dim)
        self.test_zf = torch.cat((self.test_z,f_expand),dim=2)
        self.epoch_losses = []

        self.image1 = torch.load(self.transfer_path + 'image1.sprite')
        self.image2 = torch.load(self.transfer_path + 'image2.sprite')
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
           recon_x = self.model.decode_frames(self.test_zf) 
           recon_x = recon_x.view(16,3,64,64)
           torchvision.utils.save_image(recon_x,'%s/epoch%d.png' % (self.sample_path,epoch))
    
    def recon_frame(self,epoch,original):
        with torch.no_grad():
            _,_,_,_,_,_,recon = self.model(original) 
            image = torch.cat((original,recon),dim=0)
            print(image.shape)
            image = image.view(16,3,64,64)
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
            os.makedirs(os.path.dirname('%s/epoch%d/image1_body_image2_motion.png' % (self.transfer_path,epoch)),exist_ok=True)
            torchvision.utils.save_image(image1_body_image2_motion,'%s/epoch%d/image1_body_image2_motion.png' % (transfer_path,epoch))
            torchvision.utils.save_image(image2_body_image1_motion,'%s/epoch%d/image2_body_image1_motion.png' % (transfer_path,epoch)))



    def train_model(self):
       self.model.train()
       for epoch in range(self.start_epoch,self.epochs):
           losses = []
           print("Running Epoch : {}".format(epoch+1))
           for i,data in enumerate(self.trainloader,1):
               data = data.to(device)
               self.optimizer.zero_grad()
               f_mean,f_logvar,f,z_mean,z_logvar,z,recon_x = self.model(data)
               loss = loss_fn(data,recon_x,f_mean,f_logvar,z_mean,z_logvar)
               loss.backward()
               self.optimizer.step()
               losses.append(loss.item())
           meanloss = np.mean(losses)
           self.epoch_losses.append(meanloss)
           print("Epoch {} : Average Loss: {}".format(epoch+1,meanloss))
           self.save_checkpoint(epoch)
           self.model.eval()
           self.sample_frames(epoch+1)
           sample = self.test[int(torch.randint(0,len(self.test),(1,)).item())]
           sample = torch.unsqueeze(sample,0)
           sample = sample.to(self.device)
           self.recon_frame(epoch+1,sample)
           self.style_transfer(epoch+1)
           self.model.train()
       print("Training is complete")
