import torch
import svox2
import svox2.utils
import argparse
import numpy as np
from os import path
from util.dataset import datasets
from util import config_util
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def interpolate_extrinsics(first, last):
    c2ws = [first, last]
    c2ws = np.stack(c2ws, axis=0).astype("float32")
    key_rotations = R.from_matrix(c2ws[:,:3,:3])
    key_times = range(len(c2ws))
    slerp = Slerp(key_times, key_rotations)
    c2ws_temp = slerp(np.linspace(0,len(key_times)-1,100)).as_matrix().astype("float32")   
    c2ws = np.zeros((len(c2ws_temp), 4, 4))
    c2ws[:,:3,:3] = c2ws_temp
    t1 = first[:,-1]
    t2 = last[:,-1]
    t_diff = t2-t1
    ts = np.stack([t1+t_diff*i for i in np.linspace(0,len(key_times)-1,100)], axis=0)
    c2ws[:,:,-1] = ts
    c2ws = c2ws.astype("float32")  
    return c2ws   


def render_driving(grid, dset, first_pose_name, last_pose_name, render_out_path, device):
    for i, img_name in enumerate(dset.img_files):
        if int(img_name.split(".")[0].split("_")[-1]) == first_pose_name:
            first_pose = dset.c2w[i].detach().cpu().numpy()
        elif int(img_name.split(".")[0].split("_")[-1]) == last_pose_name:
            last_pose = dset.c2w[i].detach().cpu().numpy()

    #for i in range(dset.n_images):
    #    c2ws.append(dset.c2w[i].detach().cpu().numpy())
        
    c2ws = interpolate_extrinsics(first_pose, last_pose)   
    c2ws = torch.from_numpy(c2ws).to(device=device)
    with torch.no_grad():
        n_images = c2ws.size(0)
        frames = []
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
            im.clamp_max_(1.0)
            im = im.cpu().numpy()
            im = (im * 255).astype(np.uint8)
            frames.append(im)
            im = None
        if len(frames):
            vid_path = render_out_path
            imageio.mimwrite(vid_path, frames, fps=20, macro_block_size=8)  # pip install imageio-ffmpeg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('first_pose', type=int)
    parser.add_argument('last_pose', type=int)
    config_util.define_common_args(parser)
    args = parser.parse_args()
    config_util.maybe_merge_config_file(args, allow_invalid=True)
    device = 'cuda:0'
    # Dataset
    dset = datasets[args.dataset_type](args.data_dir, split="train",
                                        **config_util.build_data_options(args))
    if not path.isfile(args.ckpt):
        args.ckpt = path.join(args.ckpt, 'ckpt.npz')
    render_out_path = path.join(path.dirname(args.ckpt), 'driving_renders')
    render_out_path += '.mp4'

    # Grid object
    grid = svox2.SparseGrid.load(args.ckpt, device=device)
    config_util.setup_render_opts(grid.opt, args)

    render_driving(grid, dset, args.first_pose, args.second_pose, render_out_path, device)

if __name__ == "__main__":
    main()
