# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:06:10 2017

@author: edward
"""
import numpy as np

## mPAS
deconv_mat_MPAS = np.array([[ 3.01382422, -1.91911423,  2.16507173],
                            [-2.73068857,  2.71631813, -1.61821771],
                            [-0.6643464 , -0.24776748,  0.70516318]], dtype=np.float32)    

## Alcian blue
deconv_mat_AB =  np.array([[-0.79418194,  1.77253377, -0.66543406],
                           [-0.71854764,  0.15075012,  1.15625095],
                           [ 1.94910038, -1.15418887,  0.2017504 ]], dtype=np.float32)

## Beta cat
deconv_mat_betaCat = np.array([[-2.72612619,  2.84373283,  0.12421618],
                               [ 1.36440217,  0.72013634, -1.36131358],
                               [ 1.95113254, -2.92114139,  1.79961455]], dtype=np.float32)

## MAOA
deconv_mat_MAOA =  np.array([[-1.4376893 ,  0.73389447,  1.57217097],
                             [ 1.99806404, -0.05742186, -1.04005587],
                             [-0.29868191,  0.7304998 , -0.61413282]], dtype=np.float32)

## STAG
# The brown is overpowering the blue on lots of slides -- cannot find nuclear halo
deconv_mat_STAG = deconv_mat_MAOA
#deconv_mat_STAG =  np.array([[0.645, 0.679, 0.35 ],
#                             [0.269, 0.568, 0.778 ],
#                             [0.633, -0.713, 0.302]], dtype=np.float32)

## NONO
deconv_mat_NONO = deconv_mat_MAOA
#deconv_mat_NONO =  np.array([[0.651, 0.701 , 0.29 ],
#                             [0.269, 0.568 , 0.778],
#                             [0.633, -0.713, 0.302]], dtype=np.float32)

## KDM6A
deconv_mat_KDM6A = deconv_mat_MAOA
#deconv_mat_KDM6A =  np.array([[-1.79694963,  1.0867945 ,  1.27533734],
#                       [ 2.31832433, -0.42953986, -0.69101626],
#                       [-0.08269765,  0.69798708, -0.71131921]], dtype=float32)


