import os
import click
import numpy as np

from glob import glob

# Reformat the flame parameters' names
reformat_dict = {"expr": "expression", "jaw_pose": "jaw_pose"}


@click.command()
@click.option("-d", "--dir", type=str, help="The directory in which the parameters need to be reformatted.")
@click.option("--ext", type=str, default=".npy", help="The file format of the paramters", show_default=True)
def reformat(dir, ext):
    filepaths = glob(os.path.join(dir, "*" + ext))
    filepaths.sort()

    for filepath in filepaths:
        params = np.load(filepath, allow_pickle=True).item()
        reformat_params = {}

        for k, v in params.items():
            if k in reformat_dict:
                new_k = reformat_dict[k]
                reformat_params[new_k] = v
            elif k == "eyes_pose":  # special
                reformat_params["leye_pose"] = v[:, :3]
                reformat_params["reye_pose"] = v[:, 3:]
            elif k == "expression":
                reformat_params[k] = v[0]
            else:
                reformat_params[k] = v

        savepath = filepath
        np.save(savepath, reformat_params)


if __name__ == "__main__":
    reformat()
