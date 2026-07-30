""" TODO: cite care pd repo for original code, changes marked with 'Adapted' """

# patch deprecated 'chumpy' package for compatibility with python 3.11+
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
    np.int = int
    np.float = float
    np.complex = complex
    np.object = object
    np.unicode = str
    np.str = str

import os
import torch
import joblib
from pathlib import Path
from tqdm import tqdm
from smplx.lbs import vertices2joints
from smplx.body_models import SMPL
from types import SimpleNamespace
import argparse
import sys

from thesis.src.care_pd.conversion_utils import (
    _DEVICE,
    generate_smpl_in_world,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


""" 
Adapted code, based on CARE-PD repo, for converting SMPL data to H36M format.
 """
def main_world_only(cfg):
    """Streamlined function to extract 3D world coordinates and skip all camera/image projections."""        
    base_name = cfg.DATA_DIR.stem

    if hasattr(cfg, 'output_filename') and cfg.output_filename is not None:
        cfg.OUT_PATH_world = cfg.OUT_PATH / cfg.output_filename
    else:
        cfg.OUT_PATH_world = cfg.OUT_PATH / f'{base_name}_h36m_3d_world.npz'
    
    h36m_regressor = torch.tensor(np.load(cfg.H36M_J_REG), dtype=torch.float32).to(_DEVICE)
    smpl_model = SMPL(model_path=cfg.MODEL_PATH, num_betas=10).to(_DEVICE)
    
    all_smpls = joblib.load(cfg.DATA_DIR)
    result_world = dict()
    
    for subject_id in tqdm(all_smpls, desc=f"Converting {base_name} to 3D World Coords"):
        for walk_id in tqdm(all_smpls[subject_id], desc=f"Processing {subject_id}"):
            smpl_data = all_smpls[subject_id][walk_id]
            if 'Trimmed' in walk_id:
                continue

            down_sample_rate = max(1, int(cfg.fps / cfg.exfps))
            
            for down in range(down_sample_rate):
                walk_name = f"{subject_id}__{walk_id}" if down_sample_rate == 1 else f"{subject_id}__{walk_id}_down{down}"
                if smpl_data['pose'].shape[0] < 30:
                    print(f"Discarding {walk_name} because it is less than 30 frames {smpl_data['pose'].shape[0]}")
                    continue
                    
                out_world, _, _ = generate_smpl_in_world(smpl_model, smpl_data, down_sample_rate, down)
                vertices_world = out_world.vertices 
                h36m_joints_world = vertices2joints(h36m_regressor, vertices_world).cpu().detach().numpy()
                
                result_world[walk_name] = h36m_joints_world

    np.savez(cfg.OUT_PATH_world, **result_world)
    return cfg.OUT_PATH_world

def convert_smpl_to_h36m(input_filename, output_dir=None, output_filename=None):
    """Wrapper for SMPL to H36M conversion.
    
    Expects input_filename to be a full path (e.g., thesis/data/processed/PD-GaM/SMPL/file.pkl)
    """
    input_path = Path(input_filename)
    cfg = SimpleNamespace()

    cfg.output_filename = output_filename
    # TODO: copy regressor and body model files over to thesis folder
    cfg.H36M_J_REG = Path('./data/preprocessing/common/J_regressor_h36m_correct.npy')
    cfg.MODEL_PATH = Path('./data/preprocessing/common/body_models/smpl/SMPL_NEUTRAL.pkl')
    cfg.DATA_DIR = input_path
    
    # Use provided output directory, or default to the relative ../../h36m path
    if output_dir:
        cfg.OUT_PATH = Path(output_dir)
    else:
        cfg.OUT_PATH = input_path.parent.parent / 'h36m' 
    
    print(f"Input Data: {cfg.DATA_DIR}")
    print(f"Output Dir: {cfg.OUT_PATH}")
    
    # HARDCODED logic for PD-GaM (change for other datasets if needed)
    cfg.db = 'PD-GaM'
    cfg.exfps = 30
    cfg.fps = 30
    
    # H36M Face Index Format
    cfg.face_joint_indx = [1, 4, 14, 11]
        
    cfg.H = 1000
    cfg.W = 1000
    
    os.makedirs(cfg.OUT_PATH, exist_ok=True)
    out_path = main_world_only(cfg)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMPL .pkl sequences to H36M .npz format.")
    parser.add_argument("-i", "--input", type=str, default="thesis/data/raw/PD-GaM/PD-GaM.pkl",
                        help="Path to the input SMPL .pkl file.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to the output directory. Defaults to two levels up + /h36m.")
    parser.add_argument("-f", "--filename", type=str, default=None, 
                        help="Optional specific output filename (e.g., 'ground_truth_3d_world.npz').")
    
    args = parser.parse_args()
    
    convert_smpl_to_h36m(input_filename=args.input, output_dir=args.output, output_filename=args.filename)