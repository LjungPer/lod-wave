# Localized orthogonal decomposition for strongly damped wave

This repository contains the code used for simulating the numerical experiments in the paper [*A generalized finite element method for the strongly damped wave equation with rapidly varying data*](https://arxiv.org/abs/2011.03311).

## Installing dependencies
Run `pip install -r requirements.txt` to install most of the dependencies.

The one exception is the `scikit-sparse` package, which was broken when this was tested.

This needs to be installed with `pip install scikit-sparse`.

`scikit-sparse` additionally requires `libsuitesparse-dev` to be installed. You can do this by running:

```sudo apt install libsuitesparse-dev```
