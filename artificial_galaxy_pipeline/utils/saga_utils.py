"""
Utility functions extracted from SAGA code base.
Taken from https://github.com/sagasurvey/saga/blob/master/SAGA/utils/functions.py

Original Author: Yao-Yuan Mao

MIT License
Copyright (c) 2017 The SAGA Survey
"""

import io
import os
import time
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from PIL import Image

__all__ = ["add_skycoord", "makedirs_if_needed", "get_decals_cutout_url", "get_decals_viewer_image"]


def add_skycoord(table, ra_label="RA", dec_label="DEC", coord_label="coord", unit="deg"):
    if coord_label not in table.colnames:
        table[coord_label] = SkyCoord(table[ra_label], table[dec_label], unit=unit)
    return table


def makedirs_if_needed(path):
    """
    Makes the directories in the path specified, if they don't exist. If they
    already exist, this returns without doing anything.
    """
    dirs = os.path.dirname(path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def get_decals_cutout_url(ra, dec, pixscale=0.262, layer="ls-dr9", size=256, use_dev=False, file_type="jpg", data_type="cutout", bands="grz"):

    file_type = str(file_type).lower()
    if file_type not in ("jpg", "fits"):
        raise ValueError("file_type must be either 'jpg' or 'fits'")

    data_type = str(data_type).lower()
    if data_type not in ("cutout", "psf", "coadd-psf"):
        raise ValueError("data_type must be either 'cutout' or 'psf'")
    asking = f"cutout.{file_type}" if data_type == "cutout" else "coadd-psf"

    if data_type != "cutout" and file_type != "fits":
        raise ValueError("Only fits files are available for psf and coadd-psf")

    extra = f"&pixscale={pixscale}&size={int(size)}" if data_type == "cutout" else ""
    dev = "-dev" if use_dev else ""

    return f"https://www.legacysurvey.org/viewer{dev}/{asking}/?ra={ra}&dec={dec}&layer={layer}&bands={bands}{extra}"


def get_decals_viewer_image(ra, dec, pixscale=0.262, layer="ls-dr9", size=256,
                            out=None, use_dev=False, timeout=60, file_type="jpg",
                            data_type="cutout", bands="grz",
                            convert_to_data=False, cache_dir=None, retry=10):
    url = get_decals_cutout_url(ra, dec, pixscale=pixscale, layer=layer, size=size, use_dev=use_dev,
                                file_type=file_type, data_type=data_type, bands=bands)
    extention = f".{file_type}"

    content = cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, url.partition("?")[2].replace("&", "_").replace("=", "") + "_" + data_type + extention)
        try:
            with open(cache_path, "rb") as f:
                content = f.read()
        except OSError:
            pass
        else:
            cache_path = None

    if content is None:
        for i in range(int(retry) + 1):
            try:
                content = requests.get(url, timeout=timeout).content
            except requests.ReadTimeout:
                time.sleep((i + 1) * 5)
            else:
                if cache_path:
                    makedirs_if_needed(cache_path)
                    with open(cache_path, "wb") as f:
                        f.write(content)
                break
        else:
            raise RuntimeError("Cannot obtain image from {}".format(url))

    if out is not None:
        if isinstance(out, str):
            if not out.lower().endswith(extention):
                out += extention
            with open(out, "wb") as f:
                f.write(content)
        else:
            out.write(content)

    if convert_to_data:
        if file_type == "fits":
            return fits.open(io.BytesIO(content))

        if file_type == "jpg":
            return Image.open(io.BytesIO(content), formats=["JPEG"])

    return content

