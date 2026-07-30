import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import os
import argparse
import pickle
import json
from pathlib import Path
from thesis.src.care_pd.visualize_skel_walk_func import (
    visualize_sequence, 
    h36m_joint_paths, 
    SMPL_joint_paths, 
    NTU_joint_paths, 
    AMASS_joint_paths
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--npzf')
    parser.add_argument('-b', '--binary')
    parser.add_argument('-npy', '--npypath')
    parser.add_argument('-f', '--format')
    parser.add_argument('-fps', '--fps', default=30, type=int)
    parser.add_argument('-p', '--projection', default='3d', type=str)
    
    # custom arguments 
    parser.add_argument('-k', '--key', default=None, type=str, help='Specify a single sequence key to visualize')
    parser.add_argument('-hs', '--heelstrikes', default=None, type=str, help='Path to the heel strikes JSON file')
    parser.add_argument('-l', '--labels', default=None, type=str, help='Path to the severity labels JSON file')
    
    args = parser.parse_args()
    print(args)
    
    if args.npzf:
        seqs = np.load(args.npzf)
        fname = Path(args.npzf).name
    elif args.binary:
        pickle_data = pickle.load(open(args.binary, 'rb'))
        seqs = {seq_name: pickle_data['pose'][i] for i,seq_name in enumerate(pickle_data['video_name'])}
        fname = Path(args.binary).name
    elif args.npypath:
        seqs = {}
        for seq_name in os.listdir(args.npypath):
            seq = np.load(os.path.join(args.npypath, seq_name))
            seqs[seq_name] = seq
        fname = Path(args.npypath).name
    else:
        raise NotImplementedError('Must supply either -b or -n option as source file path')
        
    print(f'There are {len(seqs)} sequences in the loaded file.')

    hs_data = None
    if args.heelstrikes:
        with open(args.heelstrikes, 'r') as f:
            hs_data = json.load(f)
        print("Loaded heel strike tracking data.")

    labels_data = None
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_data = json.load(f)["key_to_severity"]
        print("Loaded severity labels data.")

    if args.key:
        if args.key in seqs:
            print(f"\nShowing specific sequence: {args.key}")
            seqs = {args.key: seqs[args.key]}
        else:
            print(f"\nError: Sequence '{args.key}' not found in the dataset!")
            print(f"First 5 available keys: {list(seqs.keys())[:5]}")
            return
    else:
        print(f'The average number of frames per clip is {np.mean([len(seqs[x]) for x in seqs])}')

    for name in seqs.keys():
        if name.endswith("_frame_ids"): continue
        seq = seqs[name]
        
        seq_severity = None
        if labels_data:
            base_key = name.split('_down')[0] if '_down' in name else name
            base_key = base_key.replace('generated_walk_', '')
            if base_key in labels_data:
                seq_severity = labels_data[base_key]
            print(f"{name} | Severity Class: {seq_severity}")
        else:
            print(name)

        joint_paths = {
            'h36m': h36m_joint_paths,
            'SMPL': SMPL_joint_paths,
            'NTU': NTU_joint_paths,
            'AMASS': AMASS_joint_paths
        }
        skel_format = joint_paths[args.format]
        if fname == 'h36m_3d_world_30f_or_longer.npz' and args.projection == '2d':
            seq = seq[:, :, :2]
        if args.projection == '2d':
            invert = True
            minmax = [0, 1000, 0, 1000]
        else:
            invert = None
            minmax = None
            
        seq_hs = hs_data.get(name) if hs_data else None
            
        visualize_sequence(
            seq, 
            name + f'\n from {fname}', 
            show_joint_indexes=True, 
            joint_paths=skel_format, 
            projection=args.projection, 
            fps=args.fps, 
            invert=invert, 
            minmax=minmax, 
            save_gif=False, 
            heel_strikes=seq_hs,
            severity=seq_severity
        )
        
if __name__ == '__main__':
    main()