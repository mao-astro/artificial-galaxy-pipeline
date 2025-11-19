import os
import pandas as pd

from .mags import MagnitudeExtractor
from .model import GMM
from ..core import BackgroundImage, ImageInjector, artificial_galaxy_generator
from ..utils import makedirs_if_needed

__all__ = ["GMMBuilder"]


class GMMBuilder:
    def __init__(
        self,
        train_size=200,
        test_size=100,
        background_ra=182.5002,
        background_dec=12.5554,
        pixel_scale=0.262,
        xy_dim=901,
        data_folder="gmm_training_data",
        train_names=None,
        test_names=None,
        model_path="trained_gmm.pkl",
    ):
        self.ra = float(background_ra)
        self.dec = float(background_dec)
        self.pixel_scale = float(pixel_scale)
        self.size = int(xy_dim)
        self.data_folder = data_folder
        self.model_path = model_path

        self.train_names = [f"galaxy_train_{i:04d}" for i in range(train_size)] if train_names is None else train_names
        self.test_names = [f"galaxy_test_{i:04d}" for i in range(test_size)] if test_names is None else test_names

        self._data_table = None
        self._gmm = None

    @property
    def data_table(self):
        if self._data_table is None:
            self._check_data_availability()
            matched_tables = []
            for name in self.train_names + self.test_names:
                run_sep = MagnitudeExtractor(name, data_folder=self.data_folder)
                matched_tables.append(run_sep.matched_table)
            self._data_table = pd.concat(matched_tables, ignore_index=True)
        return self._data_table

    @property
    def gmm(self):
        if self._gmm is None:
            train_size = len(self.train_names)
            train_table, test_table = self.data_table[:train_size], self.data_table[train_size:]
            self._gmm = GMM(train_table=train_table, test_table=test_table, model_path=self.model_path)
        return self._gmm

    def _check_data_availability(self):
        missing = []
        for name in self.train_names + self.test_names:
            if not os.path.exists(f"{self.data_folder}/{name}.fits") or not os.path.exists(f"{self.data_folder}/{name}.parquet"):
                missing.append(name)

        if not missing:
            return

        background = BackgroundImage(ra=self.ra, dec=self.dec, pixel_scale=self.pixel_scale, xy_dim=self.size)
        generator = artificial_galaxy_generator(
            num_sources=len(missing), pixel_scale=self.pixel_scale, xy_dim=self.size,
        )

        makedirs_if_needed(self.data_folder + "/")
        for i, galaxy in enumerate(generator):
            galaxy.name = missing[i]
            galaxy.write_source_table(f"{self.data_folder}/{galaxy.name}.parquet")
            injector = ImageInjector(galaxy, background)
            injector.write_image_to_file(f"{self.data_folder}/{galaxy.name}.fits")
