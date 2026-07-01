# Artificial Galaxy Pipeline

A Python pipeline for generating realistic artificial ultra-faint dwarf galaxy (UFD) candidates.

## Overview

This codebase generates artificial galaxies images using **ArtPop** to inject stellar populations into DESI Legacy Survey images, and obtains the candidates photometric data using **SEP** (Source Extraction and Photometry), and a **GMM** (Gaussian Mixture Model).

## Research Context 
This project was created to support the testing and validation of current UFD search methods.  
By generating both imaging and photometric data for tens of thousands of artificial galaxies in under a day, this pipeline also enables the development and training of machine learning–based detection models.

If you use this code in research please cite [Alexis H. Brown and Yao-Yuan Mao 2025 Res. Notes AAS 9 318](https://iopscience.iop.org/article/10.3847/2515-5172/ae22ee)


## Authors and Acknowledgments

Authors: Alexis Brown, Dr. Yao-Yuan Mao

Institution: University of Utah

Built on the following open-source projects:
- [ArtPop](https://github.com/ArtificialStellarPopulations/ArtPop)
- [SEP](https://sep.readthedocs.io/en/stable/)
- [Conditional GMM](https://github.com/tmcclintock/ConditionalGMM)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Astropy](https://www.astropy.org/)
- [skimage](https://scikit-image.org/)
