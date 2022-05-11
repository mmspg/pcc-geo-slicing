"""
Point cloud block partition.
Code adapted from https://github.com/mmspg/pcc-geo-color
"""

from pyntcloud import PyntCloud
import numpy as np
import os
from tqdm import tqdm
import argparse

def partition_pc(pc_file, block_size, keep_size, normalize):
    """Partition the PC and normalize color values.
    """
    pc = PyntCloud.from_file(pc_file)
    points = pc.points

    max_range = max(points.x.max(), points.y.max(), points.z.max())
    depth = (np.floor(np.log2(max_range))+1)
    
    resolution = 2 ** depth
    
    steps = int(resolution / block_size)
    #valid_blocks = 0

    pc_blocks = []
    
    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                tmp = points[
                    ((points.x >= (i * block_size)) & (points.x < ((i + 1) * block_size))) &
                    ((points.y >= (j * block_size)) & (points.y < ((j + 1) * block_size))) &
                    ((points.z >= (k * block_size)) & (points.z < ((k + 1) * block_size)))
                ]
                # save the block if it has enough points
                if tmp.shape[0] > keep_size:
                    # move coordinates back to [0, block_size]
                    tmp.x -= i * block_size
                    tmp.y -= j * block_size
                    tmp.z -= k * block_size
                    # normalize rgb values
                    if normalize:
                        tmp.red /= 255
                        tmp.green /= 255
                        tmp.blue /= 255
                    # save the block
                    
                    pc_blocks.append((tmp.to_numpy(), (i, j, k)))

                    #new_pc = PyntCloud(tmp)
                    #new_pc.to_file(os.path.join(output_dir, f'{pc_file[:-4]}_nor_i{i:02d}j{j:02d}k{k:02d}.ply'))
                    
                    #valid_blocks += 1
                    #total_blocks += 1
                    #tqdm_handle.set_description(f'{valid_blocks} valid blocks / {total_blocks} in total')
                    
    return pc_blocks