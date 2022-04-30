# High Dynamic Range Imaging

A implementation of the whole HDR photography flow with visualization of results, including image bracketing, camera response calibration, white balance, and tone mapping.

## Introduction

Modern cameras are unable to capture the full dynamic range of commonly encountered natural scenes with limited bits that store lightness information. High-dynamic-range (HDR) photographs are generally achieved by capturing multiple standard-exposure images, often using exposure bracketing, and then merging them into a single HDR image. 

## Requirements

numpy=1.21.5
opencv=3.4.2

## Demo

See Demo.ipynb under folder /code for the usage.

## References

- [P. E. Debevec and J. Malik, “Recovering high dynamic range radiance maps from photographs,” in ACM SIGGRAPH 2008 classes, pp. 1–10, ACM, 2008.](http://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf)
