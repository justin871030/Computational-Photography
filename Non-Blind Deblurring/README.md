# Non-Blind Deblurring

A implementation of Wiener deconvolution, Richarson-Lucy (RL) deconvolution and its bilateral variant (BRL), and total variation model.

## Introduction

A slow shutter speed will introduce blurred images due to camera shake. The purpose of Non-Blind Deblurring is to restore the images through the known kernels, while avoiding the interference of noise as much as possible.

## Requirements

numpy=1.21.5
opencv=3.4.2
scipy=1.7.1
pillow=9.0.1
imageio=2.9.0
proximal=0.1.7


## Demo

See Demo.ipynb under folder /code for the analysis and learn how to deal with images by Test.ipynb.

## References

- [1] W.H.Richardson,“Bayesian-basediterativemethodofimagerestoration,”inJournaloftheopticalsocietyofAmerica,vol.62, pp. 55–59, 1972.
- [2] L.B.Rucy,“Aniterativetechniquefortherectificationofobserveddistributions,”vol.79,pp.745–754,1974.
- [3] L.L.Yuan,J.SumandH.-Y.Shum,“Progressiveinter-scaleandintra-scalenon-blindimagedeconvolution,”vol.27,pp.1–10,
2008.
- [4] F. Heide, S. Diamond, M. Nießner, J. Ragan-Kelley, W. Heidrich, and G. Wetzstein, “Proximal: Efficient image optimization
using proximal algorithms,” vol. 35, pp. 1–15, 2016.