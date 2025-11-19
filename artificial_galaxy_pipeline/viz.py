import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from easyquery import Query
from dl import queryClient as qc
from pathlib import Path
from matplotlib.patches import Rectangle, Ellipse

from .utils import add_skycoord, get_decals_viewer_image

__all__ = ["SourceData"]


class Sweep:
    bands = "GRZ"

    @classmethod
    def _add_mag_snr(cls, d):
        with np.errstate(invalid="ignore", divide="ignore"):
            # Edited 7/12
            for band in cls.bands:
                d[f"{band.lower()}_mag"] = 22.5 - 2.5 * np.log10(d[f"FLUX_{band}"] / d[f"MW_TRANSMISSION_{band}"])
                d[f"{band.lower()}_snr"] = d[f"FLUX_{band}"] * np.sqrt(d[f"FLUX_IVAR_{band}"])
            for b1, b2 in zip(cls.bands[:-1], cls.bands[1:], strict=True):
                d[f"{b1}_{b2}"] = d[f"{b1.lower()}_mag"] - d[f"{b2.lower()}_mag"]
        return d

    @staticmethod
    def _basic_cuts(d):
        d = d[d["g_mag"] - d["r_mag"] < 2]
        d = d[d["g_mag"] - d["r_mag"] > -1]
        d = d[d["PARALLAX"] * np.sqrt(d["PARALLAX_IVAR"]) < 1]
        d = d[d["SHAPE_R"] == 0]
        return d


class SourceData:
    _cols = (
        ["LS_ID", "RA", "DEC"]
        + [f"FLUX_{b}" for b in Sweep.bands]
        + [f"FLUX_IVAR_{b}" for b in Sweep.bands]
        + [f"MW_TRANSMISSION_{b}" for b in Sweep.bands]
        + [f"DCHISQ_{i}" for i in (1, 3, 5)]
        + ["SHAPE_R", "TYPE", "FITBITS", "PARALLAX", "PARALLAX_IVAR", "GAIA_PHOT_BP_MEAN_MAG"]
    )

    def __init__(
        self,
        entry=None,
        index=None,
        row_index=None,
        ra=None,
        dec=None,
        radius=None,
        name=None,
        distance=None,
        offset=(0.13, 0.13),
        source_table_name="ls_dr9.tractor",
        cutout_options=None,
        star_selection_kwargs=None,
        source_inj_table=None,
        inj_image=None,
    ) -> None:
        self._table = self._table_raw = self._has_bright_star = self.entry = None
        self.ra = self.dec = self.radius = self.name = self.distance = None
        if entry is not None:
            if isinstance(entry, Table):
                if index is not None and row_index is not None:
                    raise ValueError("You can only specify one of `index` AND `row_index`")
                if index is None and row_index is None:
                    row_index = 0
                elif index is not None:
                    if "INDEX" in entry.colnames:
                        row_index = np.flatnonzero(entry["INDEX"] == index)
                        if not len(row_index):
                            raise IndexError(f"INDEX = {index} does not exist!")
                        row_index = row_index[0]
                    else:
                        row_index = index
                self.entry = entry[int(row_index)]
            else:
                self.entry = entry
            self.ra = float(self.entry["RA"])
            self.dec = float(self.entry["DEC"])
            if "RADIUS" in entry.colnames:
                self.radius = float(self.entry["RADIUS"])
                if self.radius < 0.01:
                    self.radius = 0.01
            if "NAME" in entry.colnames:
                self.name = str(self.entry["NAME"])
            elif "INDEX" in entry.colnames:
                self.name = str(self.entry["INDEX"])
            elif row_index is not None:
                self.name = f"Row index {row_index}"
        if ra is not None:
            self.ra = float(ra)
        if dec is not None:
            self.dec = float(dec)
        if radius is not None:
            self.radius = float(radius)
        if name is not None:
            self.name = str(name)
        if distance is not None:
            self.distance = float(distance)
        if self.ra is None or self.dec is None:
            raise ValueError("Did not correctly ingest data! Cannot find RA or DEC")
        if not self.radius:
            self.radius = 0.03
        if self.name is None:
            self.name = ""
        if source_inj_table is not None:
            self._source_inj_table = source_inj_table

        self.output_fname = f"{self.name}_{self.ra:.6f}{self.dec:+.6f}"
        self.offset = tuple(offset)
        self.field_radius = np.ceil((np.hypot(*self.offset) + self.radius * 1.01) * 100) / 100
        self.source_table_name = str(source_table_name)
        self.cutout_options = dict(layer="ls-dr9", size=901)
        if cutout_options:
            self.cutout_options = dict(self.cutout_options, **cutout_options)
        self.cutout_options["file_type"] = "jpg"
        self.cutout_options["data_type"] = "cutout"
        self.cutout_options["convert_to_data"] = True
        if "pixscale" in self.cutout_options:
            del self.cutout_options["pixscale"]
        self.star_selection_kwargs = star_selection_kwargs or dict()
        self.inj_image = inj_image

    @property
    def table_raw(self):
        if self._table_raw is None:
            cols_lower = [c.lower() for c in self._cols]
            sql = f"SELECT {','.join(cols_lower)} from {self.source_table_name} WHERE q3c_radial_query(ra, dec, {self.ra}, {self.dec}, {self.field_radius});"
            t = qc.query(sql=sql, fmt="table")
            t.rename_columns(t.colnames, [c.upper() for c in t.colnames])
            t.rename_columns([f"DCHISQ_{i}" for i in (1, 3, 5)], [f"DCHISQ[{i}]" for i in (0, 2, 4)])
            self._table_raw = t
        return self._table_raw

    @property
    def table(self):
        if self._table is None:
            t = self.table_raw
            t1 = Sweep._add_mag_snr(t)
            t1 = Sweep._basic_cuts(t1)
            t["sources_basic"] = t["sources_search"] = np.isin(t["LS_ID"], t1["LS_ID"], assume_unique=True)
            t["injected"] = False
            del t1

            with np.errstate(invalid="ignore"):
                t["gr"] = t["g_mag"] - t["r_mag"]
                t["rz"] = t["r_mag"] - t["z_mag"]
            t = add_skycoord(t)

            if hasattr(self, "_source_inj_table"):
                inj = self._source_inj_table.copy()

                # Add missing columns
                inj["sources_basic"] = True
                inj["sources_search"] = True
                inj["injected"] = True  # New column to flag injected sources
                print("Number of rows in inj:", len(inj))

                inj = add_skycoord(inj)

                # Stack with the background sources
                t = vstack([t, inj], join_type="outer")

            if self.radius:
                distance = t["coord"].separation(SkyCoord(self.ra, self.dec, unit="deg")).deg
                t["mask"] = distance < self.radius
                t["mask_offset"] = t["coord"].separation(SkyCoord(self.ra + self.offset[0], self.dec + self.offset[1], unit="deg")).deg < self.radius
                t["mask_annulus"] = (distance >= self.radius * 2) & (distance < self.radius * 3)

            self._table = t
        return self._table

    def plot_image(self, ax, zoom=1.1):
        half_side_deg = zoom * self.radius
        #   pixscale = half_side_deg * 2 * 3600 / self.cutout_options["size"]
        pixscale = half_side_deg * 2 * 3600 / 0.262
        if self.inj_image is None:
            im = get_decals_viewer_image(self.ra, self.dec, pixscale=pixscale, **self.cutout_options)
        else:
            im = self.inj_image
        ra1 = self.ra - half_side_deg / np.cos(np.deg2rad(self.dec))
        ra2 = self.ra + half_side_deg / np.cos(np.deg2rad(self.dec))
        dec1 = self.dec - half_side_deg
        dec2 = self.dec + half_side_deg
        if not hasattr(ax, "__iter__"):
            ax = [ax]
        for ax_this in ax:
            width = 2 * self.radius / np.cos(np.deg2rad(self.dec))
            height = 2 * self.radius
            circ = Ellipse((self.ra, self.dec), width=width, height=height, fill=False, edgecolor="white", lw=1.5)
            ax_this.add_patch(circ)
            if self.inj_image is None:
                ax_this.imshow(im, extent=[ra2, ra1, dec1, dec2], aspect="auto")
            else:
                ax_this.imshow(im, extent=[ra2, ra1, dec1, dec2], aspect="auto", origin="lower")

            ax_this.ticklabel_format(useOffset=False)
            ax_this.set(xlabel="R.A.", ylabel="Dec.")
        half_side_deg = zoom * self.radius
        return ra2, ra1, dec1, dec2

    def plot_hex(self, ax, xlim=(-1, 2), ylim=(25.5, 18.5), gridsize=(12, 22)):
        d = self.table
        weight = d["mask"].astype(np.float64) - d["mask_annulus"].astype(np.float64) / 5
        dx = abs(xlim[1] - xlim[0]) * 0.01
        dy = abs(ylim[1] - ylim[0]) * 0.02
        if not hasattr(ax, "__iter__"):
            ax = [ax]
        for ax_this in ax:
            ax_this.hexbin(
                d["gr"],
                d["r_mag"],
                C=weight,
                reduce_C_function=np.sum,
                gridsize=gridsize,
                cmap="RdBu_r",
                vmax=4,
                vmin=-4,
                extent=(xlim[0] - dx, xlim[1] + dx, ylim[1] - dy, ylim[0] + dy),
            )
            ax_this.set_xlim(*xlim)
            ax_this.set_ylim(*ylim)
            ax_this.set_xlabel("$g-r$")

    def get_filtered_table(self):
        d = self.table
        source_mask = d["sources_search"]
        return d[source_mask]

    def make_plot_minimal(self, save_to=None, sources_shown="search", show_basic_sources=True, adjustment_function=None, show=True, title=None):
        fig, ax = plt.subplot_mosaic(
            [["image_zoom", "cmd_o"], ["stars", "cmd_o"]],
            figsize=(9, 10),
            constrained_layout=False,
            width_ratios=[1.0, 0.8],
            height_ratios=[1.0, 1.0],
            dpi=96,
            gridspec_kw=dict(wspace=0.25, hspace=0.35),
        )

        d = self.table

        mask_color_color = Query("gr < 1.3", "gr > -0.2").mask(d)
        mask_c0 = d["mask_offset"]
        mask_c1 = d["mask"] & mask_color_color
        mask_c3 = d["mask"] & (~mask_color_color)

        if sources_shown == "search" and show_basic_sources:
            mask_c1_small = mask_c1 & d["sources_basic"] & (~d["sources_search"])
            mask_c3_small = mask_c3 & d["sources_basic"] & (~d["sources_search"])
        else:
            show_basic_sources = False

        if sources_shown in ["search", "basic"]:
            source_mask = d[f"sources_{sources_shown}"]
            mask_c1 = mask_c1 & source_mask
            mask_c0 = mask_c0 & source_mask
            mask_c3 = mask_c3 & source_mask
        elif sources_shown == "all":
            pass
        else:
            raise ValueError("`sources_shown` must be 'all', 'basic', or 'search'.")

        for k in ("image_zoom", "stars"):
            ax[k].set_box_aspect(1)

        ra2, ra1, dec1, dec2 = self.plot_image([ax["image_zoom"]])

        ax_this = ax["stars"]
        if d["RA"].max() - d["RA"].min() > 180:
            ra_ = np.where(d["RA"] > 180, d["RA"] - 360, d["RA"])
        else:
            ra_ = d["RA"]
        ax_this.scatter(ra_[source_mask], d["DEC"][source_mask], s=1, c="C2", edgecolors="None")
        ax_this.set(xlabel="R.A.", ylabel="Dec.")
        xlim = ax_this.get_xlim()
        ax_this.set_xlim(xlim[::-1])

        x0 = min(ra1, ra2)
        y0 = dec1
        width = max(ra1, ra2) - x0
        height = dec2 - dec1
        ax_this.add_patch(Rectangle((x0, y0), width, height, fill=False, lw=1.5))

        ax_this = ax["cmd_o"]
        ax_this.scatter(d["gr"][mask_c1], d["r_mag"][mask_c1], s=20, color="C1", edgecolors="None")
        ax_this.scatter(d["gr"][mask_c3], d["r_mag"][mask_c3], s=20, color="C1", edgecolors="None", alpha=0.4)

        if show_basic_sources:
            ax_this.scatter(d["gr"][mask_c1_small], d["r_mag"][mask_c1_small], s=5, color="C1", edgecolors="None")
            ax_this.scatter(d["gr"][mask_c3_small], d["r_mag"][mask_c3_small], s=5, color="C1", edgecolors="None", alpha=0.4)
        ax_this.set_xlim(-1, 2)
        ax_this.set_ylim(25.5, 18.5)
        ax_this.set(xlabel="$g-r$", ylabel="$r$")
        ax_this.text(-0.75, 19, f"N = {np.count_nonzero(mask_c1 | mask_c3)}")

        if title:
            fig.text(0.5, 0.96, title, ha="center", va="top", fontsize=18)
            fig.text(0.5, 0.92, f"({self.ra:.6f}, {self.dec:+.6f})", ha="center", va="top", fontsize=16)

        if adjustment_function is not None:
            adjustment_function(fig, ax)

        if save_to:
            Path(save_to).mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{save_to}/{self.output_fname}_minimal.jpg", pil_kwargs=dict(quality=120))

        if show:
            plt.show()
            return None
        return fig
