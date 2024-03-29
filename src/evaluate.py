import pandas as pd
from pyntcloud import PyntCloud
import os

"""
Objective metrics computation.
Code adapted from https://github.com/mmspg/pcc-geo-color
and https://github.com/mmspg/learned-residual-pcc
"""

def evaluate_pc(ori_path, dec_path, nor_path, resolution, pc_error_path):
    """
    Evaluates the reconstructed point clouds quality based on D1 and D2 PSNR.
    Parameters:
        ori_path: Path to the original uncompressed evaluated point cloud .
        dec_path: path to the reconstructed point cloud evaluated.
        nor_path: Path to the normals of the evaluated point cloud.
                  Used to calculate the D2 metrics.
        resolution: Resolution of the point cloud.
        pc_error_path: path to the MPEG binary

    Output: D1 PSNR, D2 PSNR.
    """

    pd.set_option('mode.chained_assignment', None)   
    p2po_psnr = p2pl_psnr = c0_psnrf = c1_psnrf = c2_psnrf = None

    if os.path.split(pc_error_path)[1] == pc_error_path:
        pc_error_path = f"./{pc_error_path}"
        
    os.system(f'{pc_error_path} -a {ori_path} -b {dec_path} -n {nor_path} --color=0 --resolution={resolution} > tmp.log')
    with open('tmp.log', 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            if ('mseF,PSNR' in line) and ('(p2point):' in line):
                p2po_psnr = float(line.split(':')[-1])
            elif ('mseF,PSNR' in line) and ('(p2plane):' in line):
                p2pl_psnr = float(line.split(':')[-1])
        
        if p2po_psnr is not None:
            g_metric_D1 = p2po_psnr
        else:
            g_metric_D1 = -1
        
        if p2pl_psnr is not None:
            g_metric_D2 = p2pl_psnr
        else:
            g_metric_D2 = -1
            

    return g_metric_D1, g_metric_D2
