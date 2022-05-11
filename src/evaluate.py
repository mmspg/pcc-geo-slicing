import pandas as pd
from pyntcloud import PyntCloud
import os

"""
Objective metrics computation.
Code adapted from https://github.com/mmspg/pcc-geo-color
and https://github.com/mmspg/learned-residual-pcc
"""

def evaluate_pc(ori_path, dec_path, nor_path, resolution):
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

    Output: D1 PSNR, D2 PSNR, one color metric per channel and one global
    color metric.
    """

    pd.set_option('mode.chained_assignment', None)   
    p2po_psnr = p2pl_psnr = c0_psnrf = c1_psnrf = c2_psnrf = None
    os.system(f'./pc_error -a {ori_path} -b {dec_path} -n {nor_path} --color=0 --resolution={resolution} > tmp.log')
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
