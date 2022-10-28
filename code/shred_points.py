# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \SHRED_GEDI\code\shred_points.py
###   @Author: AceSix
###   @Date: 2022-10-27 20:54:44
###   @LastEditors: AceSix
###   @LastEditTime: 2022-10-27 21:00:07
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################
# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \SHRED_GEDI\code\shred_mesh.py
###   @Author: AceSix
###   @Date: 2022-10-27 12:35:11
###   @LastEditors: AceSix
###   @LastEditTime: 2022-10-27 16:42:18
###   @Copyright (C) 2022 Brown U. All rights reserved.
###################################################################
import torch
import sys, os
from methods.srd import SRD
import utils
import numpy as np

NORMALIZE = True

class Shape:
    def __init__(self, mesh_name):
        points = []
        with open(mesh_name) as f:
            lines = f.readlines()
            for line in lines:
                segs = line.strip().split(',')
                point = np.array(list(map(float, segs[:3])))
                points.append(np.expand_dims(point, 0))
        verts = np.concatenate(points, axis = 0)
        verts = torch.tensor(verts).float().to(utils.device)

        if NORMALIZE:
            center = (verts.max(dim=0).values + verts.max(dim=0).values) / 2.
            verts -= center.unsqueeze(0)
            scale = verts.norm(dim=1).max()
            verts /= scale
        
        self.verts = verts
        self.points = None
    

def shred_mesh(mesh_name, out_name):    
    shape = Shape(mesh_name)

    M = SRD()
    srd_points, srd_regions = M.make_pred(shape)

    utils.vis_pc(srd_points, srd_regions, out_name)
        
    
if __name__ == '__main__':
    # shred_mesh(sys.argv[1], sys.argv[2])
    shred_mesh('G:/Brown/Courses/CSCI1230-22F/Scenefiles/mesh/teapot.obj', './code/results/teapot_out')

    
