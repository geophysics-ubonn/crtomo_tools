#!/bin/bash

# test plots for complex data in sipdir
cd sipdir_complex
sd_plot2
cp *Magnitude.png ../mag_cplx1.png
cp *Phase.png ../phase_cplx1.png
cp *Real\ Part.png ../real_cplx1.png
cp *Imaginary\ Part.png ../imag_cplx1.png
rm *.png
sd_plot2 -x -10 -X 40 -z -10 -Z 0 
cp *Magnitude.png ../mag_cplx2.png
cp *Phase.png ../phase_cplx2.png
cp *Real\ Part.png ../real_cplx2.png
cp *Imaginary\ Part.png ../imag_cplx2.png
rm *.png
sd_plot2 -u cm --alpha_cov --cmaglin 
cp *Magnitude.png ../mag_cplx3.png
cp *Phase.png ../phase_cplx3.png
cp *Real\ Part.png ../real_cplx3.png
cp *Imaginary\ Part.png ../imag_cplx3.png
rm *.png
sd_plot2 --cbtiks 5 
cp *Magnitude.png ../mag_cplx4.png
cp *Phase.png ../phase_cplx4.png
cp *Real\ Part.png ../real_cplx4.png
cp *Imaginary\ Part.png ../imag_cplx4.png
rm *.png
sd_plot2 --mag_vmin 1.5 --mag_vmax 2.5 --imag_vmin -7 --imag_vmax -4 --pha_vmin -3 --pha_vmax 0 --real_vmin -2 --real_vmax 0
cp *Magnitude.png ../mag_cplx5.png
cp *Phase.png ../phase_cplx5.png
cp *Real\ Part.png ../real_cplx5.png
cp *Imaginary\ Part.png ../imag_cplx5.png
rm *.png
rm *.png

# test plots for FPI data in sipdir
cd ../sipdir_fpi
sd_plot2 --title
cp *Magnitude.png ../mag_fpi1.png
cp *Phase.png ../phase_fpi1.png
cp *Real\ Part.png ../real_fpi1.png
cp *Imaginary\ Part.png ../imag_fpi1.png
rm *.png
sd_plot2 -x -10 -X 40 -z -10 -Z 0 
cp *Magnitude.png ../mag_fpi2.png
cp *Phase.png ../phase_fpi2.png
cp *Real\ Part.png ../real_fpi2.png
cp *Imaginary\ Part.png ../imag_fpi2.png
rm *.png
sd_plot2 -u cm --alpha_cov --cmaglin 
cp *Magnitude.png ../mag_fpi3.png
cp *Phase.png ../phase_fpi3.png
cp *Real\ Part.png ../real_fpi3.png
cp *Imaginary\ Part.png ../imag_fpi3.png
rm *.png
sd_plot2 --cbtiks 5
cp *Magnitude.png ../mag_fpi4.png
cp *Phase.png ../phase_fpi4.png
cp *Real\ Part.png ../real_fpi4.png
cp *Imaginary\ Part.png ../imag_fpi4.png
rm *.png
sd_plot2 --mag_vmin 1.5 --mag_vmax 2.5 --imag_vmin -7 --imag_vmax -4 --pha_vmin -3 --pha_vmax 0 --real_vmin -2 --real_vmax 0
cp *Magnitude.png ../mag_fpi5.png
cp *Phase.png ../phase_fpi5.png
cp *Real\ Part.png ../real_fpi5.png
cp *Imaginary\ Part.png ../imag_fpi5.png
rm *.png

# test plots for DC data in sipdir
cd ../sipdir_mag
sd_plot2 --title
cp *.png ../mag_dc1.png
rm *.png
sd_plot2 -x -10 -X 40 -z -10 -Z 0 
cp *.png ../mag_dc2.png
rm *.png
sd_plot2 -u cm --alpha_cov --cmaglin
cp *.png ../mag_dc3.png
rm *.png
sd_plot2 --cbtiks 5 
cp *.png ../mag_dc4.png
rm *.png
sd_plot2 --mag_vmin 1.5 --mag_vmax 2.5 --imag_vmin -7 --imag_vmax -4 --pha_vmin -3 --pha_vmax 0 --real_vmin -2 --real_vmax 0
cp *.png ../mag_dc5.png
rm *.png
