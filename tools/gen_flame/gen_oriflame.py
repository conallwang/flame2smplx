import sys, os

import click
import torch
import numpy as np

from flame2023.flame import FlameHead
from tqdm import tqdm
from glob import glob


def write_obj(filepath, verts, tris=None, log=True):
    """save obj files

    Args:
        verts:      Vx3, vertices coordinates
        tris:       n_facex3, faces consisting of vertices id
    """
    fw = open(filepath, "w")
    # vertices
    for vert in verts:
        fw.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

    if not tris is None:
        for tri in tris:
            fw.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
    fw.close()
    if log:
        print(f"mesh has been saved in {filepath}.")


@click.command()
@click.option("-d", "--processed_dir", help="The directory that saves processed FLAME params.")
def gen_flame_objs(processed_dir):
    flame_cfgs = {
        "flame.n_shape": 300,
        "flame.n_expr": 100,
        "flame.n_pose": 15,
        "flame.add_teeth": False,
        "flame.model_path": "./assets/flame2023.pkl",
        "flame.lmk_embedding_path": "./assets/landmark_embedding_with_eyes.npy",
    }
    flame = FlameHead(flame_cfgs).cuda()

    outdir = os.path.join(os.path.dirname(processed_dir), "example_obj")
    os.makedirs(outdir, exist_ok=True)

    processed_list = glob(os.path.join(processed_dir, "*.npy"))
    processed_list.sort()

    bar = tqdm(range(len(processed_list)), desc="----")
    for params_path in processed_list:
        filename = os.path.basename(params_path).replace("npy", "obj")

        bar.set_description(desc=filename)
        flame_param = np.load(params_path, allow_pickle=True).item()

        shape_params = torch.from_numpy(flame_param["shape"][None]).cuda()
        expr_params = torch.from_numpy(flame_param["expression"][None]).cuda()
        rotation_params = torch.from_numpy(flame_param["rotation"]).cuda()
        neck_params = torch.from_numpy(flame_param["neck_pose"]).cuda()
        jaw_params = torch.from_numpy(flame_param["jaw_pose"]).cuda()
        eyes_params = torch.from_numpy(
            np.concatenate([flame_param["leye_pose"], flame_param["reye_pose"]], axis=-1)
        ).cuda()
        translation_params = torch.from_numpy(flame_param["translation"]).cuda()
        # static_offset = torch.from_numpy(flame_param["static_offset"]).cuda()[:, :5023]

        head_verts, head_lmks = flame(
            shape=shape_params,
            expr=expr_params,
            rotation=rotation_params,
            neck=neck_params,
            jaw=jaw_params,
            eyes=eyes_params,
            translation=translation_params,
            zero_centered_at_root_node=True,
            static_offset=None,
        )
        head_faces = flame.faces[:9976]

        savepath = os.path.join(outdir, filename)
        write_obj(savepath, head_verts[0], head_faces + 1, log=False)

        bar.update()


if __name__ == "__main__":
    gen_flame_objs()
