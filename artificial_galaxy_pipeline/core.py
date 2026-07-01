import numpy as np

from astropy import units as u
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from scipy.stats import logistic
from skimage.transform import rescale

import artpop

from .utils import sdss_rgb, get_decals_viewer_image


__all__ = ["ArtificialGalaxy", "BackgroundImage", "ImageInjector", "artificial_galaxy_generator"]

NATIVE_PIXEL_SCALE = 0.262

def zoom_image(image_dict, zoom_factor):
    for band in image_dict:
        H, W = [round(s * zoom_factor) for s in image_dict[band].shape]
        image_dict[band] = rescale(image_dict[band], zoom_factor, preserve_range=True)
        #print(f"New Image Shape for {band}: {image_dict[band].shape}")
    return image_dict
    
class ArtificialGalaxy:
    def __init__(
        self,
        total_mass,
        distance,
        scale_radius=None,
        log_age=9.8,
        feh=-1.9,
        phot_system="DECam",
        pixel_scale=NATIVE_PIXEL_SCALE,
        xy_dim=901,
        random_seed=None,
        name=None,
        **kwargs,
    ):
        self.name = name or "galaxy"
        self.pixel_scale = float(pixel_scale)
        self.xy_dim = int(xy_dim)
        self.phot_system = str(phot_system)
        self.rng = np.random.default_rng(random_seed)

        self.artpop_source_factory = artpop.MISTPlummerSSP

        if scale_radius is None:
            scale_radius = 1.39 * (total_mass**0.4) * u.pc

        self.artpop_source_props = dict(
            random_state=np.random.RandomState(self.rng.integers(1000000)),
            log_age=float(log_age),
            feh=float(feh),
            scale_radius=scale_radius,
            total_mass=total_mass,
            distance=distance,
            pixel_scale=self.pixel_scale,
            phot_system=self.phot_system,
            xy_dim=self.xy_dim,
            **kwargs,
        )

        self._artpop_source = self._source_table = None

    @property
    def artpop_source(self):
        if self._artpop_source is None:
            self._artpop_source = self.artpop_source_factory(**self.artpop_source_props)
        return self._artpop_source

    def get_coords_on_sky(self, center_ra=None, center_dec=None):
        """
        Convert image pixel coordinates (x, y) into sky coordinates (RA, Dec)
        using the WCS projection defined by the object's center and pixel scale.
        """
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = np.array([(self.xy_dim + 1) / 2, (self.xy_dim + 1) / 2])
        w.wcs.cdelt = np.array([-self.pixel_scale / 3600, self.pixel_scale / 3600])
        w.wcs.crval = np.array([center_ra, center_dec])
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return w.wcs_pix2world(self.artpop_source.x, self.artpop_source.y, 0)  # the 0 is strange but it must be there

    def get_observed_gr_magnitudes(self, gmm, detection_adjustment=(24.1, 0.22), retention_fraction=0.95):
        g_true = np.array(self.artpop_source.mags[f"{self.phot_system}_g"])
        r_true = np.array(self.artpop_source.mags[f"{self.phot_system}_r"])

        detection_prob = logistic.sf(r_true, *detection_adjustment) * retention_fraction
        mask = self.rng.random(len(r_true)) < detection_prob

        g_true = g_true[mask]
        r_true = r_true[mask]

        new_seed = self.rng.integers(1000000)
        current_random_state = np.random.get_state()
        np.random.seed(new_seed)

        dmag_r, dmag_g = np.fromiter(
            (gmm.sample_conditional_joint(ri, gi, n_samples=1) for ri, gi in zip(r_true, g_true)), dtype=np.dtype((np.float64, 2)), count=len(r_true)
        ).T

        np.random.set_state(current_random_state)

        g_obs = g_true + dmag_g
        r_obs = r_true + dmag_r

        return g_obs, r_obs, mask

    @property
    def source_table(self):
        if self._source_table is None:
            src = self.artpop_source
            t = Table(
                {
                    "x": src.x,
                    "y": src.y,
                    "g_mag": src.mags[f"{self.phot_system}_g"],
                    "r_mag": src.mags[f"{self.phot_system}_r"],
                }
            )
            t["gr"] = t["g_mag"] - t["r_mag"]
            t["name"] = self.name
            t["pixel_scale"] = self.pixel_scale
            t["size"] = self.xy_dim
            self._source_table = t
        return self._source_table

    def get_observed_source_table(self, gmm=None, center_ra=None, center_dec=None):
        t = self.source_table.copy()
        mask = None
        if gmm is not None:
            g_obs, r_obs, mask = self.get_observed_gr_magnitudes(gmm)
            t = t[mask]
            t["g_mag"] = g_obs
            t["r_mag"] = r_obs
            t["gr"] = g_obs - r_obs

        if center_ra is not None and center_dec is not None:
            ra, dec = self.get_coords_on_sky(center_ra, center_dec)
            if mask is not None:
                ra, dec = ra[mask], dec[mask]
            t["RA"] = ra
            t["DEC"] = dec

        return t

    def write_source_table(self, filepath):
        self.source_table.write(filepath, overwrite=True)


class BackgroundImage:
    def __init__(self, ra, dec, pixel_scale=NATIVE_PIXEL_SCALE, xy_dim=901, layer="ls-dr9", cache_dir="image_cache", zoom_factor=None):
        self.ra = ra
        self.dec = dec
        self.pixel_scale = pixel_scale
        self.xy_dim = xy_dim
        self.layer = layer
        self.cache_dir = cache_dir
        self.zoom_factor = zoom_factor

        self._psf = self._coadd = self._rgb = None
        

    def fetch_psf(self):
        """
        Returns PSFs in dictionary with keys 'g', 'r', and 'z'.
        """
        hdulist = get_decals_viewer_image(
            self.ra,
            self.dec,
            pixscale=self.pixel_scale,
            layer=self.layer,
            size=self.xy_dim,
            file_type="fits",
            data_type="psf",
            convert_to_data=True,
            cache_dir=self.cache_dir,
        )
        psf = {band: hdulist[i].data for i, band in enumerate("grz")}

        if self.pixel_scale != NATIVE_PIXEL_SCALE:
            psf = zoom_image(psf, NATIVE_PIXEL_SCALE / self.pixel_scale)
            
        return psf

    def fetch_coadd(self):
        """
        Fetches coadds (cutout images) of the sky for a given region specified by the params.
        @param: based on the right ascension ra and declination dec
        @return: the image corresponding to the three photometric bands: g, r, z
        """
        cutout = get_decals_viewer_image(
            self.ra, self.dec, pixscale=self.pixel_scale, layer=self.layer, size=self.xy_dim, file_type="fits", convert_to_data=True, cache_dir=self.cache_dir
        )
        cutout = cutout[0].data
        image = {band: cutout[i, :, :] for i, band in enumerate("grz")}

        if self.zoom_factor is not None:
            image = zoom_image(image, self.zoom_factor)
            
        return image

    @property
    def psf(self):
        if self._psf is None:
            self._psf = self.fetch_psf()
        return self._psf

    @property
    def coadd(self):
        if self._coadd is None:
            self._coadd = self.fetch_coadd()
        return self._coadd

    @property
    def rgb(self):
        if self._rgb is None:
            bands = "zrg"
            self._rgb = sdss_rgb(*[self.coadd[b] for b in bands], bands=bands)
        return self._rgb


class ImageInjector:
    def __init__(self, artificial_galaxy=None, background_image=None, center_adjustment=(0, 0), zoom_factor=None, verbose=True):
        self.artificial_galaxy = artificial_galaxy
        self.background_image = background_image
        self.center_adjustment = np.array(center_adjustment, dtype=int)
        self.verbose = bool(verbose)
        self.zoom_factor = zoom_factor

        assert artificial_galaxy is not None or background_image is not None, "At least one of artificial_galaxy or background_image must be provided."
        assert self.center_adjustment.shape == (2,), "center_adjustment must be a tuple of two integers."
        
        if artificial_galaxy is not None and background_image is not None:
            assert artificial_galaxy.pixel_scale == background_image.pixel_scale, "Pixel scales of artificial_galaxy and background_image must match."
            if artificial_galaxy.pixel_scale != NATIVE_PIXEL_SCALE:
                print("Warning: Injection should be done at native pixel scale.")
        
        self.zpt = 22.5
        self.imager = artpop.IdealImager()
        self._source_image = self._injected_image = self._rgb = None

    @property
    def source_image(self):
        if self._source_image is None:
            self._source_image = {}
            for b in "grz":
                if self.background_image is None:
                    psf = artpop.moffat_psf(fwhm=1.2 * u.arcsec)
                else:
                    psf = self.background_image.psf[b]

                self._source_image[b] = self.imager.observe(
                    self.artificial_galaxy.artpop_source, bandpass=f"{self.artificial_galaxy.phot_system}_{b}", psf=psf, zpt=self.zpt
                ).image
        return self._source_image

    @property
    def injected_image(self):
        if self._injected_image is None:
            if self.artificial_galaxy is None:  # No injection, just return background
                return self.background_image.coadd

            if self.background_image is None:  # Empty background
                size = self.artificial_galaxy.xy_dim
                bkg = {b: np.zeros((size, size)) for b in "grz"}
            else:
                size = self.background_image.xy_dim
                bkg = self.background_image.coadd

            center = self.center_adjustment + size // 2
            if self.verbose:
                print(f"Injecting {self.artificial_galaxy.name} at position {center}")

            src = self.source_image
            bkg_slice, src_slice = artpop.util.embed_slices(center, bkg["g"].shape, src["g"].shape)
            self._injected_image = {b: (bkg[b][bkg_slice] + src[b][src_slice]) for b in "grz"}

            if self.zoom_factor is not None:
                self._injected_image = zoom_image(self._injected_image, self.zoom_factor)

        return self._injected_image

    @property
    def rgb(self):
        if self._rgb is None:
            bands = "zrg"
            self._rgb = sdss_rgb(*[self.injected_image[b] for b in bands], bands=bands)
        return self._rgb

    def write_image_to_file(self, filepath):
        """Write a single FITS file with image data and header parameters."""

        hdu = fits.PrimaryHDU(data=np.stack(list(self.injected_image.values())))

        if self.artificial_galaxy is not None:
            hdr = hdu.header
            for key, val in self.artificial_galaxy.artpop_source_props.items():
                key = key[:8].upper()
                if hasattr(val, "value") and hasattr(val, "unit"):
                    hdr[key] = (val.value, str(val.unit))
                elif isinstance(val, (int, float)):
                    hdr[key] = val
                else:
                    hdr[key] = str(val)

        hdu.writeto(filepath, overwrite=True)
        
def artificial_galaxy_generator(
    num_sources=1,
    log_mass_range=(4, 6),
    log_age_range=(10, 10.2),
    feh_range=(-2.2, -1.7),
    distance_range=(0.4, 1.2),
    random_seed=None,
    name_prefix=None,
    **kwargs,
):
    rng = np.random.default_rng(random_seed)
    if not num_sources:
        num_sources = 1

    name_prefix = str(name_prefix or "galaxy")
    d = int(np.floor(np.log10(num_sources))) + 1

    for i in range(num_sources):
        total_mass = 10 ** rng.uniform(*log_mass_range)
        log_age = rng.uniform(*log_age_range)
        feh = rng.uniform(*feh_range)
        distance = rng.uniform(*distance_range) * u.Mpc

        yield ArtificialGalaxy(
            total_mass=total_mass,
            log_age=log_age,
            feh=feh,
            distance=distance,
            random_seed=rng.integers(1000000),
            name=f"{name_prefix}_{i + 1:0{d}d}",
            **kwargs,
        )

 