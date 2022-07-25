import torch
import svox2
import svox2.utils
import argparse
import numpy as np
from os import path
from util.dataset import datasets
from util.util import pose_spherical
from util import config_util
import imageio
from tqdm import tqdm


def render_circle_func(grid, dset, render_out_path, traj_type, device):
    # View params
    offset = np.array([0,0,0]) # center
    elevation = -45.0
    elevation2 = -12 # for spiral
    num_views = 600
    radius = 2.00
    up_rot = dset.c2w[:, :3, :3].cpu().numpy()
    ups = np.matmul(up_rot, np.array([0, -1.0, 0])[None, :, None])[..., 0]
    vec_up = np.mean(ups, axis=0)
    vec_up /= np.linalg.norm(vec_up)
    # Generate poses 
    if traj_type == 'spiral':
        angles = np.linspace(-180, 180, num_views + 1)[:-1]
        elevations = np.linspace(elevation, elevation2, num_views)
        c2ws = [
            pose_spherical(
                angle,
                ele,
                radius,
                offset,
                vec_up=vec_up,
            )
            for ele, angle in zip(elevations, angles)
        ]
        c2ws += [
            pose_spherical(
                angle,
                ele,
                radius,
                offset,
                vec_up=vec_up,
            )
            for ele, angle in zip(reversed(elevations), angles)
        ]
    else :
        c2ws = [
            pose_spherical(
                angle,
                elevation,
                radius,
                offset,
                vec_up=vec_up,
            )
            for angle in np.linspace(-180, 180, num_views + 1)[:-1]
        ]
    c2ws = np.stack(c2ws, axis=0)
    c2ws = torch.from_numpy(c2ws).to(device=device)
    with torch.no_grad():
        n_images = c2ws.size(0)
        frames = []
        grid.opt.near_clip = 0.0
        dset_w = dset.get_image_size(0)[1]
        dset_h = dset.get_image_size(0)[0]
        for img_id in tqdm(range(0, n_images)):
            cam = svox2.Camera(c2ws[img_id],
                            dset.intrins.get('fx', 0),
                            dset.intrins.get('fy', 0),
                            dset.intrins.get('cx', img_id), # dset_w * 0.5
                            dset.intrins.get('cy', img_id), # dset_h * 0.5
                            dset_w, dset_h,
                            ndc_coeffs=(-1.0, -1.0))
            torch.cuda.synchronize()
            im = grid.volume_render_image(cam, use_kernel=True)
            torch.cuda.synchronize()
            #im.clamp_(0.0, 1.0)
            im.clamp_max_(1.0)
            im = im.cpu().numpy()
            im = (im * 255).astype(np.uint8)
            frames.append(im)
            im = None
        if len(frames):
            vid_path = render_out_path
            imageio.mimwrite(vid_path, frames, fps=30, macro_block_size=8)  # pip install imageio-ffmpeg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    config_util.define_common_args(parser)
    parser.add_argument('--traj_type',
                        choices=['spiral', 'circle'],
                        default='spiral',
                        help="Render a spiral (doubles length, using 2 elevations), or just a cirle")
    args = parser.parse_args()
    config_util.maybe_merge_config_file(args, allow_invalid=True)
    device = 'cuda:0'
    # Dataset
    dset = datasets[args.dataset_type](args.data_dir, split="test",
                                        **config_util.build_data_options(args))
    if not path.isfile(args.ckpt):
        args.ckpt = path.join(args.ckpt, 'ckpt.npz')
    render_out_path = path.join(path.dirname(args.ckpt), 'circle_renders')
    render_out_path += '.mp4'

    # Grid object
    grid = svox2.SparseGrid.load(args.ckpt, device=device)
    config_util.setup_render_opts(grid.opt, args)
    render_circle_func(grid, dset, render_out_path, args.traj_type, device)

if __name__ == "__main__":
    main()
