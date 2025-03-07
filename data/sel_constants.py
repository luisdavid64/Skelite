"""
/*====================================================================*
 -  Copyright (C) 2001 Leptonica.  All rights reserved.
 -
 -  Redistribution and use in source and binary forms, with or without
 -  modification, are permitted provided that the following conditions
 -  are met:
 -  1. Redistributions of source code must retain the above copyright
 -     notice, this list of conditions and the following disclaimer.
 -  2. Redistributions in binary form must reproduce the above
 -     copyright notice, this list of conditions and the following
 -     disclaimer in the documentation and/or other materials
 -     provided with the distribution.
 -
 -  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 -  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 -  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 -  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL ANY
 -  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 -  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 -  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 -  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 -  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 -  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 -  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *====================================================================*/
"""

import torch

sel_4_1 = torch.tensor([
    [ 0.,  0.,  1.],
    [-1.,  0.,  1.],
    [ 0.,  0.,  1.]
], dtype=torch.float32)


sel_4_2 = torch.tensor([
    [ 0.,  0.,  1.],
    [-1.,  0.,  1.],
    [ 0., -1.,  0.]
], dtype=torch.float32)

sel_4_3 = torch.tensor([
    [ 0., -1.,  0.],
    [-1.,  0.,  1.],
    [ 0.,  0.,  1.]
], dtype=torch.float32)

sel_8_2 = torch.tensor([
    [  0,  1,  0],
    [ -1,  0,  1],
    [ -1,  0,  0]
], dtype=torch.float32)



def generate_rotations(tensor):
    rot90 = torch.rot90(tensor, 1, [0, 1]).contiguous()
    rot180 = torch.rot90(tensor, 2, [0, 1]).contiguous()
    rot270 = torch.rot90(tensor, 3, [0, 1]).contiguous()
    return [tensor, rot90, rot180, rot270]

"""Structuring elements + Isomorphic rotations"""
SEL_LIST = [ 
    *generate_rotations(sel_4_2),
    *generate_rotations(sel_4_3),
]

def generate_hit_miss(sel, is_torch=True):
    hit = torch.zeros_like(sel, dtype=torch.float32)
    miss = torch.zeros_like(sel, dtype=torch.float32)
    hit[sel == 1] = 1
    miss[sel == -1] = 1
    if is_torch:
        hit = hit.unsqueeze(0).unsqueeze(0)
        miss = miss.unsqueeze(0).unsqueeze(0)
    
    return hit, miss

SEL_LIST = [generate_hit_miss(sel) for sel in SEL_LIST]