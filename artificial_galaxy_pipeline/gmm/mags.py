from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropy.io import fits
from astropy.table import Table
import sep

__all__ = ["MagnitudeExtractor"]


class MagnitudeExtractor:
    def __init__(self, galaxy_name, data_folder, bkg_subtraction=False):
        self._galaxy_name = galaxy_name
        self._data_folder = data_folder
        self._data = {}
        self._bkg = {}
        self._objects = {}
        self._matched_source_table = None

        self.bands = {"g": 0, "r": 1}

        self.true_mags = None
        self.bkg_subtraction = bkg_subtraction
        self.matched_indices = None
        self._check_required_files()

    def _check_required_files(self):
        fits_ok = True
        missing = []

        fits_path = Path(f"{self._data_folder}/{self.galaxy_name}.fits")
        if not fits_path.exists():
            fits_ok = False
            missing.append(str(fits_path))

        mag_path = Path(f"{self._data_folder}/{self.galaxy_name}.parquet")
        if not mag_path.exists():
            fits_ok = False
            missing.append(str(mag_path))

        if not fits_ok:
            missing_list = "\n".join(missing)
            raise FileNotFoundError(f"Missing required files for galaxy '{self.galaxy_name}':\n{missing_list}")

    @property
    def galaxy_name(self):
        if self._galaxy_name is None:
            raise ValueError("A galaxy name must be provided to run SEP")
        return self._galaxy_name

    @property
    def matched_table(self):
        if self._matched_source_table is None:
            self._matched_source_table = self.injection_matching()
        return self._matched_source_table

    @property
    def objects(self):
        if not self._objects:
            for band in self.bands:
                data = (self.get_data(band) - self.get_bkg(band)) if self.bkg_subtraction else self.get_data(band)
                self._objects[band] = sep.extract(data, 1.5, err=self.get_bkg(band).globalrms)

        return self._objects

    def get_data(self, band):
        if self._data.get(band) is None:
            data = self.read_fits_file()
            for band, index in self.bands.items():
                self._data[band] = data[index]
        return self._data[band]

    def get_bkg(self, band):
        if self._bkg.get(band) is None:
            self._bkg[band] = self.estimate_background(band)
        return self._bkg[band]

    def estimate_background(self, band):
        band_bkg = sep.Background(self.get_data(band))
        return band_bkg

    def read_fits_file(self):
        with fits.open(f"{self._data_folder}/{self.galaxy_name}.fits") as hdul:
            fits_data = hdul[0].data
            fits_data = fits_data.astype(fits_data.dtype.newbyteorder("="))
        return fits_data

    def injection_matching(self):
        true_mags = Table.read(f"{self._data_folder}/{self.galaxy_name}.parquet")

        bright_mask = (true_mags["r_mag"] < 26) & (true_mags["g_mag"] < 26)
        true_mags = true_mags[bright_mask]

        try:
            r_mags = self.aperture_photometry(band="r")
            g_mags = self.aperture_photometry(band="g")
            matched_objects = self.compute_r_band_matching(r_mags, g_mags, true_mags)

            self.matched_indices = matched_objects["detected_idx"]

            matched_table = pd.DataFrame(
                {
                    "x": r_mags["x"][self.matched_indices],
                    "y": r_mags["y"][self.matched_indices],
                    "r_mag": r_mags["mag"][self.matched_indices],
                    "g_mag": g_mags["mag"][self.matched_indices],
                    "true_mag_r": matched_objects["true_mag_r"].values,
                    "dmag_r": matched_objects["dmag_r"].values,
                    "true_mag_g": matched_objects["true_mag_g"].values,
                    "dmag_g": matched_objects["dmag_g"].values,
                    "sep": matched_objects["sep"].values,
                }
            )

            return matched_table

        except Exception as e:
            print(f"[{self.galaxy_name}] Skipping due to zero matches / detections : {e}")
            return pd.DataFrame()

    def compute_r_band_matching(self, r_mags, g_mags, true_mags):
        points1 = np.column_stack((true_mags["x"], true_mags["y"]))
        points2 = np.column_stack((r_mags["x"], r_mags["y"]))

        kd_tree1 = cKDTree(points1)
        kd_tree2 = cKDTree(points2)

        indices = kd_tree1.query_ball_tree(kd_tree2, r=2.83)

        indices = kd_tree1.query_ball_tree(kd_tree2, r=2.83)

        idx1 = np.concatenate([[i] * len(idx) for i, idx in enumerate(indices)])
        idx2 = np.concatenate(indices)
        idx1 = idx1.astype(int)
        idx2 = idx2.astype(int)
        sep = np.sqrt(((points1[idx1] - points2[idx2]) ** 2).sum(axis=1))  # Computes distance between 2 points
        dmag_r = r_mags["mag"][idx2] - true_mags["r_mag"][idx1]
        dmag_g = g_mags["mag"][idx2] - true_mags["g_mag"][idx1]

        match_df = pd.DataFrame(
            {
                "true_idx": idx1,
                "detected_idx": idx2,
                "sep": sep,
                "true_mag_r": true_mags["r_mag"][idx1],
                "true_mag_g": true_mags["g_mag"][idx1],
                "dmag_r": dmag_r,
                "dmag_g": dmag_g,
            }
        )

        # Step 1: Filter by magnitude difference
        match_df = match_df[(match_df["dmag_r"].abs() < 1.5) & (match_df["dmag_g"].abs() < 1.5)]

        # Step 2: Sort by distance
        match_df = match_df.sort_values(by="sep")

        # Step 3: Drop duplicates, keeping closest (smallest seperation) per detection
        unique_matches = match_df.drop_duplicates(subset="detected_idx", keep="first")

        return unique_matches

    def aperture_photometry(self, band):
        objects = self.objects["r"]
        valid = ~np.isnan(objects["x"]) & ~np.isnan(objects["y"]) & ~np.isnan(objects["a"]) & ~np.isnan(objects["b"]) & (objects["a"] > 0) & (objects["b"] > 0)
        objects = objects[valid]
        objects_params = {k: objects[k] for k in ("x", "y", "a", "b", "theta")}

        flux, fluxerr, flag = sep.sum_ellipse(data=self.get_data(band), r=3.0, err=self.get_bkg(band).globalrms, gain=1.0, **objects_params)
        with np.errstate(divide="ignore", invalid="ignore"):
            mags = 22.5 - 2.5 * np.log10(flux)
        return Table({"x": objects["x"], "y": objects["y"], "mag": mags})

    def plot_detections(self):
        r_obj = self.objects["r"]
        g_obj = self.objects["g"]

        if self.matched_indices is None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 21))

            self.show_data(ax=axes[0, 0], band="r", add_objects=False)
            axes[0, 0].set_title(f"{self.galaxy_name} R Band")

            self.show_data(ax=axes[0, 1], band="r", add_objects=True, objects=r_obj)
            axes[0, 1].set_title(f"All Detections R Band")

            self.show_data(ax=axes[1, 0], band="g", add_objects=False)
            axes[1, 0].set_title(f"G Band")

            self.show_data(ax=axes[1, 1], band="g", add_objects=True, objects=g_obj)
            axes[1, 1].set_title(f"All Detections G Band")
            plt.tight_layout()
            plt.show()
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 21))

        r_filtered_obj = r_obj[self.matched_indices]
        g_filtered_obj = g_obj[self.matched_indices]

        self.show_data(ax=axes[0, 0], band="g", add_objects=False)
        axes[0, 0].set_title(f"{self.galaxy_name} R Band")

        self.show_data(ax=axes[0, 1], band="r", add_objects=True, objects=r_obj)
        axes[0, 1].set_title(f"All Detections R Band")

        self.show_data(ax=axes[0, 2], band="r", add_objects=True, objects=r_filtered_obj)
        axes[0, 2].set_title(f"Postion Matched Detections R Band")

        self.show_data(ax=axes[1, 0], band="g", add_objects=False)
        axes[1, 0].set_title(f"{self.galaxy_name} G Band")

        self.show_data(ax=axes[1, 1], band="g", add_objects=True, objects=g_obj)
        axes[1, 1].set_title(f"All Detections G Band")

        self.show_data(ax=axes[1, 2], band="g", add_objects=True, objects=g_filtered_obj)
        axes[1, 2].set_title(f"Position Matched Detections G Band")

        plt.tight_layout()
        plt.show()

    def show_data(self, band, ax, objects=None, add_objects=False):
        data = (self.get_data(band) - self.get_bkg(band)) if self.bkg_subtraction else self.get_data(band)
        m, s = np.mean(data), np.std(data)

        ax.imshow(data, interpolation="nearest", cmap="gray", vmin=m - s, vmax=m + s, origin="lower")

        if add_objects:
            self.draw_object_ellipses(ax, objects)

    def draw_object_ellipses(self, ax, objects, color="red"):
        objects = objects[
            (~np.isnan(objects["x"]))
            & (~np.isnan(objects["y"]))
            & (~np.isnan(objects["a"]))
            & (~np.isnan(objects["b"]))
            & (objects["a"] > 0)
            & (objects["b"] > 0)
        ]
        for obj in objects:
            e = Ellipse(
                xy=(obj["x"], obj["y"]),
                width=6 * obj["a"],
                height=6 * obj["b"],
                angle=obj["theta"] * 180.0 / np.pi,
                edgecolor=color,
                facecolor="none",
                linewidth=1.5,
            )
            ax.add_artist(e)
