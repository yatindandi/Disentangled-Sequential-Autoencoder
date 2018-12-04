from tqdm import *
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim


class Sprites(data.Dataset):
    def __init__(self, path, size):
        self.path = path
        self.length = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = torch.load(self.path+'/%d.sprite' % (idx+1))
        return item['body'], item['shirt'], item['pant'], item['hair'], item['action'], item['sprite']


class SpriteClassifier(nn.Module):
    def __init__(self, n_bodies=7, n_shirts=4, n_pants=5, n_hairstyles=6, n_actions=3,
                 num_frames=8, in_size=64, channels=64, code_dim=1024, hidden_dim=512, nonlinearity=None):
        super(SpriteClassifier, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        encoding_conv = []
        encoding_conv.append(nn.Sequential(nn.Conv2d(3, channels, 5, 4, 1, bias=False), nl))
        size = in_size // 4
        self.num_frames = num_frames
        while size > 4:
            encoding_conv.append(nn.Sequential(
                nn.Conv2d(channels, channels * 2, 5, 4, 1, bias=False),
                nn.BatchNorm2d(channels * 2), nl))
            size = size // 4
            channels *= 2
        self.encoding_conv = nn.Sequential(*encoding_conv)
        self.final_size = size
        self.final_channels = channels
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.encoding_fc = nn.Sequential(
                nn.Linear(size * size * channels, code_dim),
                nn.BatchNorm1d(code_dim), nl)
        # The last hidden state of a convolutional LSTM over the scenes is used for classification
        self.classifier_lstm = nn.LSTM(code_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.body = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2), nl,
                nn.Linear(hidden_dim // 2, n_bodies))
        self.shirt = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2), nl,
                nn.Linear(hidden_dim // 2, n_shirts))
        self.pants = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2), nl,
                nn.Linear(hidden_dim // 2, n_pants))
        self.hairstyles = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2), nl,
                nn.Linear(hidden_dim // 2, n_hairstyles))
        self.action = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2), nl,
                nn.Linear(hidden_dim // 2, n_actions))

    def forward(self, x):
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.encoding_conv(x)
        x = x.view(-1, self.final_channels * (self.final_size ** 2))
        x = self.encoding_fc(x)
        x = x.view(-1, self.num_frames, self.code_dim)
        # Classifier output depends on last layer of LSTM: Can also change this to a bi-LSTM if required
        _, (hidden, _) = self.classifier_lstm(x)
        hidden = hidden.view(-1, self.hidden_dim)
        return self.body(hidden), self.shirt(hidden), self.pants(hidden), self.hairstyles(hidden), self.action(hidden)


def save_model(model, optim, epoch, path):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict()}, path)

def check_accuracy(model, test, device):
    total = 0
    correct_body = 0
    correct_shirt = 0
    correct_pant = 0
    correct_hair = 0
    correct_action = 0
    with torch.no_grad():
        for item in test:
            body, shirt, pant, hair, action, image = item
            image = image.to(device)
            body = body.to(device)
            shirt = shirt.to(device)
            pant = pant.to(device)
            hair = hair.to(device)
            action = action.to(device)
            pred_body, pred_shirt, pred_pant, pred_hair, pred_action = model(image)
            _, pred_body = torch.max(pred_body.data, 1)
            _, pred_shirt = torch.max(pred_shirt.data, 1)
            _, pred_pant = torch.max(pred_pant.data, 1)
            _, pred_hair = torch.max(pred_hair.data, 1)
            _, pred_action = torch.max(pred_action.data, 1)
            total += body.size(0)
            correct_body += (pred_body == body).sum().item()
            correct_shirt += (pred_shirt == shirt).sum().item()
            correct_pant += (pred_pant == pant).sum().item()
            correct_hair += (pred_hair == hair).sum().item()
            correct_action += (pred_action == action).sum().item()
    print('Accuracy, Body : {} Shirt : {} Pant : {} Hair : {} Action {}'.format(correct_body/total, correct_shirt/total, correct_pant/total, correct_hair/total, correct_action/total)) 


def train_classifier(model, optim, dataset, device, epochs, path, test, start=0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start, epochs):
        running_loss = 0.0
        for i, item in tqdm(enumerate(dataset, 1)):
            body, shirt, pant, hair, action, image = item
            image = image.to(device)
            body = body.to(device)
            shirt = shirt.to(device)
            pant = pant.to(device)
            hair = hair.to(device)
            action = action.to(device)
            pred_body, pred_shirt, pred_pant, pred_hair, pred_action = model(image)
            loss = criterion(pred_body, body) + criterion(pred_shirt, shirt) + criterion(pred_pant, pant) + criterion(pred_hair, hair) + criterion(pred_action, action)
            loss.backward()
            optim.step()
            running_loss += loss.item()
        print('Epoch {} Avg Loss {}'.format(epoch + 1, running_loss / i))
        save_model(model, optim, epoch, path)
        check_accuracy(model, test, device)

device = torch.device('cuda:0')
model = SpriteClassifier()
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.0003)
sprites_train = Sprites('./dataset/lpc-dataset/train', 6759)
sprites_test = Sprites('./dataset/lpc-dataset/test', 801)
loader = data.DataLoader(sprites_train, batch_size=32, shuffle=True, num_workers=4)
loader_test = data.DataLoader(sprites_test, batch_size=64, shuffle=True, num_workers=4)
train_classifier(model, optim, loader, device, 20, './checkpoint_classifier.pth', loader_test) 
