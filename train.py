import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
num_epochs = 1000
batch_size = 32
learning_rate = 1e-5
input_size = 2048 * 3
hidden_size = 400
latent_size = 20
class_choice = "guitar"

modelnet40 = Dataset(os.path.abspath("datasets/"), split="all", class_choice=class_choice, random_jitter=False, random_rotate=False, random_translate=False)
train_data = DataLoader(modelnet40, batch_size=batch_size, shuffle=True, pin_memory=True)
vae = VAE(input_size, hidden_size, latent_size, input_size).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def loss_fn(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KL = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
    return BCE + KL


for epoch in range(1, num_epochs + 1):
    # Resume
    if osp.exists(osp.sep.join(("checkpoints", str(epoch) + ".pth"))):
        temp = torch.load(osp.sep.join(("checkpoints/", str(epoch) + ".pth")))
        ret = vae.load_state_dict(temp['state_dict'])
        print(ret)
        optimizer.load_state_dict(temp['optimizer'])
        # scheduler.load_state_dict(temp['scheduler'])
        continue

    for data in train_data:
        inputs, _, _,  _ = data
        inputs = inputs.view(-1, input_size).to(device)
        optimizer.zero_grad()
        recon_inputs, mu, log_var = vae(inputs)

        # Loss
        loss = loss_fn(recon_inputs, inputs, mu, log_var)
        loss.backward()
        optimizer.step()

    # scheduler.step() 
    print("Epoch {:<4} Loss: {:<8.4f}".format(epoch, loss.item()))
    checkpoints = {
        'state_dict': vae.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
    }
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    torch.save(checkpoints, osp.sep.join(("checkpoints", str(epoch) + ".pth")))