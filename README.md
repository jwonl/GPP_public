# Radiation Image Reconstruction and Uncertainty Quantification Using a Gaussian Process Prior

This repository contains example code for the paper, "Radiation Image Reconstruction and Uncertainty Quantification Using a Gaussian Process Prior." The package ```imager``` can be used for image reconstruction and uncertainty quantification methods presented in the paper. The included notebook demonstrates an example measurement scenario where an anisotropic Gaussian source and a ring-shaped source distribution are reconstructed in a single detector mapping.

## Installation
To install in an editable mode, use 
```bash
pip install -e .
```

## Usage
The package ```imager``` contains sevaral modules useful for single detector mapping problems. 
1. ```detector.py``` contains a class, ```SphericalDetector``` for simulating a sinlge spherical detector (i.e., isotropic response) with a custom intrinsic efficiency.
2. ```path.py``` include two classes: ```WalkingPath``` and ```RasterPath``` for random walking path and raster path simulation.
3. ```source.py``` for various distributed source classes, such as ```GaussianSource``` and ```RingGaussianSource```.
4. ```scenario.py``` is used to hold a detector, a path and source objects. Then a ```Scenario``` object computes the system matrix of a measurement sceanrio. A sceanario object can also be used to plot a grount truth source distribution and count measurement.
5. ```imager.py``` contains the ```Imager``` class, which is primarily for image reconstruction and uncertainty quantification. The class provides the maximum likelihood expectation maximization (MLEM) and Gaussian Process Prior (GPP) image reconstruction methods. A The Bayesian uncertainty quantification using the Laplace approximation and preconditioned Crank-Nikolson MCMC.
6. ```utilities.py``` provides some useful functions.

The notebook file ```notebooks/example_1.ipynb``` shows how these different classes can be set up and used for image reconstruction and uncertainty quantificaiton. 

