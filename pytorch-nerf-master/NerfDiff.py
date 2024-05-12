import torch
from torch import optim
import numpy as np
from torch import nn

from run_pixelnerf_alt import load_data, set_up_test_data, PixelNeRF
from conditional_diff import DDPM, ContextUnet  

class NerfDiff(nn.Module):
    def __init__(self, pixelnerf, device):
        super(NerfDiff, self).__init__()
        self.pixelnerf = pixelnerf
        # Assume ContextUnet is the neural network model used within DDPM
        self.diffusion_model = DDPM(nn_model=ContextUnet(in_channels=3),  # Adjust in_channels as needed
                                    betas=(1e-4, 0.02),  # Example betas
                                    n_T=400,  # Number of timesteps
                                    device=device)
        self.device = device

    def forward(self, ds, os, source_image):
        # Generate initial predictions using PixelNeRF
        initial_output = self.pixelnerf(ds, os, source_image)
        C_rs_c, C_rs_f = initial_output
        C_rs_f = C_rs_f.reshape(3, 12, 12)
        C_rs_c = C_rs_c.reshape(3, 12, 12)
        print(f"Shape of C_rs_f: {C_rs_f.shape}")
        print(f"Shape of C_rs_c: {C_rs_c.shape}")

        C_rs_f = self.diffusion_model.forward(C_rs_f,C_rs_c)

        return C_rs_c, C_rs_f
    
def main():
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda:0"

    (num_iters, train_dataset) = load_data()
    img_size = train_dataset[0][2].shape[0]

    pixelnerf = PixelNeRF(device, train_dataset.camera_distance, train_dataset.scale)
    nerf_diff = NerfDiff(pixelnerf, device).to(device)

    batch_img_size = 12
    n_batch_pix = batch_img_size**2
    n_objs = 4

    lr = 1e-4
    train_params = list(nerf_diff.pixelnerf.F_c.parameters()) + list(nerf_diff.pixelnerf.F_f.parameters())+list(nerf_diff.diffusion_model.parameters())
    optimizer = optim.Adam(train_params, lr=lr)
    criterion = torch.nn.MSELoss()

    (test_source_image, test_R, test_target_image) = set_up_test_data(train_dataset, device)
    init_o = train_dataset.init_o.to(device)
    init_ds = train_dataset.init_ds.to(device)
    test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
    test_os = (test_R @ init_o).expand(test_ds.shape)

    psnrs = []
    iternums = []
    use_bbox = True
    num_bbox_iters = 300000
    display_every = 100

    nerf_diff.pixelnerf.F_c.train()
    nerf_diff.pixelnerf.F_f.train()
    nerf_diff.pixelnerf.E.eval()
    nerf_diff.diffusion_model.train()

    for i in range(num_iters):
        if i == num_bbox_iters:
            use_bbox = False

        loss = 0
        for obj in range(n_objs):
            try:
                (source_image, R, target_image, bbox) = train_dataset[0]
            except ValueError:
                continue

            R = R.to(device)
            ds = torch.einsum("ij,hwj->hwi", R, init_ds)
            os = (R @ init_o).expand(ds.shape)

            if use_bbox:
                pix_rows = np.arange(bbox[0], bbox[2])
                pix_cols = np.arange(bbox[1], bbox[3])
            else:
                pix_rows = np.arange(0, img_size)
                pix_cols = np.arange(0, img_size)

            pix_row_cols = np.meshgrid(pix_rows, pix_cols, indexing="ij")
            pix_row_cols = np.stack(pix_row_cols).transpose(1, 2, 0).reshape(-1, 2)
            choices = np.arange(len(pix_row_cols))
            try:
                selected_pix = np.random.choice(choices, n_batch_pix, False)
            except ValueError:
                continue

            pix_idx_rows = pix_row_cols[selected_pix, 0]
            pix_idx_cols = pix_row_cols[selected_pix, 1]
            ds_batch = ds[pix_idx_rows, pix_idx_cols].reshape(batch_img_size, batch_img_size, -1)
            os_batch = os[pix_idx_rows, pix_idx_cols].reshape(batch_img_size, batch_img_size, -1)

            (C_rs_c, C_rs_f) = nerf_diff(ds_batch, os_batch, source_image)
            target_img = target_image.to(device)
            target_img_batch = target_img[pix_idx_rows, pix_idx_cols].reshape(C_rs_c.shape)
            loss += criterion(C_rs_c, target_img_batch)
            loss += criterion(C_rs_f, target_img_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
