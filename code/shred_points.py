# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \SHRED_GEDI\code\shred_points.py
###   @Author: AceSix
###   @Date: 2022-10-27 20:54:44
###   @LastEditors: AceSix
###   @LastEditTime: 2022-11-08 11:20:25
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
    shred_mesh('G:/GEDI_data/LaSelva_txt/La_Selva_05102019_001_825900_1154700.txt', 
               './code/LaSelva_results/La_Selva_05102019_001_825900_1154700')
               
    shred_mesh('G:/GEDI_data/LaSelva_txt/La_Selva_05102019_001_825900_1154800.txt', 
               './code/LaSelva_results/La_Selva_05102019_001_825900_1154800')
               
    shred_mesh('G:/GEDI_data/LaSelva_txt/La_Selva_05102019_001_825900_1154900.txt', 
               './code/LaSelva_results/La_Selva_05102019_001_825900_1154900')
               
    shred_mesh('G:/GEDI_data/LaSelva_txt/La_Selva_05102019_001_825900_1155000.txt', 
               './code/LaSelva_results/La_Selva_05102019_001_825900_1155000')
               
    shred_mesh('G:/GEDI_data/LaSelva_txt/La_Selva_05102019_001_825900_1155100.txt', 
               './code/LaSelva_results/La_Selva_05102019_001_825900_1155100')
               
    shred_mesh('G:/GEDI_data/LaSelva_txt/La_Selva_05102019_001_825900_1155200.txt', 
               './code/LaSelva_results/La_Selva_05102019_001_825900_1155200')

    
