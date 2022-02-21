import pandas as pd
from pyntcloud import PyntCloud
import os

"""
Block merging and algorithm evaluation.
Code adapted from https://github.com/mmspg/pcc-geo-color
and https://github.com/mmspg/learned-residual-pcc
"""

def merge_pc(ori_file, div_files, block_size, div_dir, output_dir):
    """
    Merges the decompressed blocks of a PC, to reconstruct
    the entire PC.
    
    Parameters:
        ori_file: Directory with original unpartitioned point clouds.
       div_files: List of decompressed files.
      block_size: Shape of the cubic block (eg 128 if 128x128x128).
         div_dir: Directory containing the decompressed files.
      output_dir: Output directory for reconstructed point clouds.
                      
    Output: Saves the reconstructed point clouds from the decompressed
            blocks into output_dir.
    """
            
    cur_div_files = [f for f in div_files if ori_file[:-4] in f]
    
    total_pieces = len(cur_div_files)
    points = pd.DataFrame(data={ 'x': [], 'y': [], 'z': [] })
    
    for div_file in cur_div_files:
        div_pc = PyntCloud.from_file(os.path.join(div_dir, div_file))
        div_pc_points = div_pc.points
        ind = [int(div_file.split('_')[-1][1:3]), int(div_file.split('_')[-1][4:6]), int(div_file.split('_')[-1][7:9])]
        div_pc_points.x += ind[0] * block_size
        div_pc_points.y += ind[1] * block_size
        div_pc_points.z += ind[2] * block_size
        
        points = pd.concat([points, div_pc_points])
        
    points.reset_index(drop=True, inplace=True)
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    res_pc = PyntCloud(points)
    res_pc.to_file(os.path.join(output_dir, f'{ori_file[:-4]}_dec.ply'))


def evaluate_pc(ori_path, dec_path, nor_path, resolution, colorspace):
    """
    Evaluates the reconstructed point clouds quality based on the D1,
    D2 and various color metrics. The pc_error utility
    from mpeg needs to exist and to be in the same directory as the 
    code that calls this function.
    
    Parameters:
        ori_path: Path to the original uncompressed evaluated point cloud .
        dec_path: path to the reconstructed point cloud evaluated.
        nor_path: Path to the normals of the evaluated point cloud.
                  Used to calculate the D2 metrics.
      resolution: Resolution of the point cloud.
      colorspace: Either 1 for yuv, or 0 for rgb. Irrelevant in this
                  case for geometry compression.

    Output: D1 PSNR, D2 PSNR, one color metric per channel and one global
    color metric.
    """

    pd.set_option('mode.chained_assignment', None)   
    p2po_psnr = p2pl_psnr = c0_psnrf = c1_psnrf = c2_psnrf = None
    os.system(f'./pc_error -a {ori_path} -b {dec_path} -n {nor_path} --color=1 --resolution={resolution} --mseSpace={colorspace} > tmp.log')
    with open('tmp.log', 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            if ('mseF,PSNR' in line) and ('(p2point):' in line):
                p2po_psnr = float(line.split(':')[-1])
            elif ('mseF,PSNR' in line) and ('(p2plane):' in line):
                p2pl_psnr = float(line.split(':')[-1])
            elif ('c[0],PSNRF' in line) and ('h.' not in line):
                c0_psnrf = float(line.split(':')[-1])
            elif ('c[1],PSNRF' in line) and ('h.' not in line):
                c1_psnrf = float(line.split(':')[-1])
            elif ('c[2],PSNRF' in line) and ('h.' not in line):
                c2_psnrf = float(line.split(':')[-1])    
        
        if p2po_psnr is not None:
            g_metric_D1 = p2po_psnr
        else:
            g_metric_D1 = -1
        
        if p2pl_psnr is not None:
            g_metric_D2 = p2pl_psnr
        else:
            g_metric_D2 = -1
            
        if c0_psnrf and c1_psnrf and c2_psnrf is not None:
            c_metric_ch0 = c0_psnrf
            c_metric_ch1 = c1_psnrf
            c_metric_ch2 = c2_psnrf
            if colorspace == 1:
                c_metric = (6 * c0_psnrf + c1_psnrf + c2_psnrf) / 8
            else:
                c_metric = (c0_psnrf + c1_psnrf + c2_psnrf) / 3
        else:
            c_metric_ch0 = c_metric_ch1 = c_metric_ch2 = c_metric = -1

    return g_metric_D1, g_metric_D2, c_metric_ch0, c_metric_ch1, c_metric_ch2, c_metric
