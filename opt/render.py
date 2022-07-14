import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import compute_ssim
from util import config_util

import imageio
from tqdm import tqdm


def render_func(grid, dset, render_dir, device, write_vid):
    with torch.no_grad():
        n_images = dset.n_images
        avg_psnr = 0.0
        avg_ssim = 0.0
        frames = []
        for img_id in tqdm(range(0, n_images)):
            c2w = dset.c2w[img_id].to(device=device)
            dset_h, dset_w = dset.get_image_size(img_id)
            cam = svox2.Camera(c2w,
                            dset.intrins.get('fx', img_id),
                            dset.intrins.get('fy', img_id),
                            dset.intrins.get('cx', img_id),
                            dset.intrins.get('cy', img_id),
                            dset_w, dset_h,
                            ndc_coeffs=dset.ndc_coeffs)
            im = grid.volume_render_image(cam, use_kernel=True)
            #im.clamp_(0.0, 1.0)
            im.clamp_max_(1.0)
            im_gt = dset.gt[img_id].to(device=device)
            mse = (im - im_gt) ** 2
            mse_num : float = mse.mean().item()
            psnr = -10.0 * math.log10(mse_num)
            avg_psnr += psnr
            ssim = compute_ssim(im_gt, im).item()
            avg_ssim += ssim
            print(img_id, 'PSNR', psnr, 'SSIM', ssim)
            img_path = path.join(render_dir, f'{img_id:04d}.png');
            im = im.cpu().numpy()
            im_gt = dset.gt[img_id].numpy()
            im = np.concatenate([im_gt, im], axis=1)
            im = (im * 255).astype(np.uint8)
            imageio.imwrite(img_path,im)
            frames.append(im)
            im = None
        if len(frames) and write_vid:
            vid_path = render_dir + '.mp4'
            imageio.mimwrite(vid_path, frames, fps=30, macro_block_size=8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    config_util.define_common_args(parser)
    parser.add_argument('--train', action='store_true', default=False, help='render train set')
    args = parser.parse_args()
    config_util.maybe_merge_config_file(args, allow_invalid=True)
    device = 'cuda:0'
    if not path.isfile(args.ckpt):
        args.ckpt = path.join(args.ckpt, 'ckpt.npz')
    render_dir = path.join(path.dirname(args.ckpt),
                'train_renders' if args.train else 'test_renders')
    os.makedirs(render_dir, exist_ok=True)
    dset = datasets[args.dataset_type](args.data_dir, split="test_train" if args.train else "test",
                                        **config_util.build_data_options(args))
    grid = svox2.SparseGrid.load(args.ckpt, device=device)
    config_util.setup_render_opts(grid.opt, args)
    render_func(grid, dset, render_dir, device, True)

if __name__ == "__main__":
    main()
