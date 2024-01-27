import torch
import os.path as osp
import open3d as o3d

from model.vae import VAE

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


checkpoint = osp.sep.join(("checkpoints", "31.pth"))
input_size = 2048 * 3
hidden_size = 400
latent_size = 20

vae = VAE(input_size, hidden_size, latent_size, input_size).to(device)

with torch.no_grad():
    vae.eval()
    temp = torch.load(checkpoint, map_location=device)
    ret = vae.load_state_dict(temp['state_dict'])
    print(ret)

    for i in range(10):
        latent_sample = torch.randn(latent_size).to(device)
        generated_data = vae.decoder(latent_sample).view(2048, 3).cpu()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(generated_data.numpy())
        o3d.visualization.draw_geometries([pcd])