import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path as osp

from model.vae import VAE
from data.dataset import Dataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

'''
Parameters
'''
parser = argparse.ArgumentParser()
# Overall options
parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--input_size", default=2048 * 3, help="Input image's size (W x H)")
parser.add_argument("--hidden_size", type=int, default=400, help="Hidden layer's dimenion")
parser.add_argument("--latent_size", type=int, default=20, help="Latent Layer's dimension")

opt = parser.parse_args()

modelnet40 = Dataset(os.path.abspath("datasets/"), split="all")
train_data = DataLoader(modelnet40, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
vae = VAE(opt.input_size, opt.hidden_size, opt.latent_size, opt.input_size).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=opt.learning_rate)

def loss_fn(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KL = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
    return BCE + KL


for epoch in range(1, opt.num_epochs + 1):
    # Resume
    if osp.exists(osp.sep.join(("checkpoints", str(epoch) + ".pth"))):
        temp = torch.load(osp.sep.join(("checkpoints/", str(epoch) + ".pth")))
        ret = vae.load_state_dict(temp['state_dict'])
        print(ret)
        optimizer.load_state_dict(temp['optimizer'])
        continue

    for data in train_data:
        inputs, _, _,  _ = data
        inputs = inputs.view(-1, opt.input_size).to(device)
        optimizer.zero_grad()
        recon_inputs, mu, log_var = vae(inputs)

        # Loss
        loss = loss_fn(recon_inputs, inputs, mu, log_var)
        loss.backward()
        optimizer.step()
    
    print("Epoch {:<4} Loss: {:<8.4f}".format(epoch, loss.item()))
    checkpoints = {
        'state_dict': vae.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoints, osp.sep.join(("checkpoints", str(epoch) + ".pth")))