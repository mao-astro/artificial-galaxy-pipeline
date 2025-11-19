"""
Utility functions extracted from legacypipe code base.
Taken from https://github.com/legacysurvey/legacypipe/blob/main/py/legacypipe/survey.py

Original Author: Dustin Lang

This source file is licensed under the BSD 3-Clause License.

Copyright 2015-2018 LegacyPipe Contributors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np

def sdss_rgb(*imgs, bands="grz", scales=None, m=0.03, Q=20, mnmx=None, clip=True, rgb_stretch_factor=1):

    rgbscales = dict(
       g =    (2, 6.0 * rgb_stretch_factor),
       r =    (1, 3.4 * rgb_stretch_factor),
       i =    (0, 3.0 * rgb_stretch_factor),
       z =    (0, 2.2 * rgb_stretch_factor),
    )

    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img, band in zip(imgs, bands):
        _, scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
    if Q is not None:
        fI = np.arcsinh(Q * I) / np.sqrt(Q)
        I += (I == 0.) * 1e-6
        I = fI / I
    H, W = I.shape
    rgb = np.zeros((H,W,3), np.float32)

    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        if mnmx is None:
            imgplane = (img * scale + m) * I
        else:
            mn,mx = mnmx
            imgplane = ((img * scale + m) - mn) / (mx - mn)
        if clip:
            imgplane = np.clip(imgplane, 0, 1)
        rgb[:,:,plane] = imgplane

    return rgb
