#!/bin/bash

# test plots for complex data in sipdir
cd sipdir_complex
sd_plot 
cp *.png ../mag_cplx1.png
rm *.png
sd_plot -x -10 -X 40 -z -10 -Z 0 
cp *.png ../mag_cplx2.png
rm *.png
sd_plot -u cm --alpha_cov --cmaglin 
cp *.png ../mag_cplx3.png
rm *.png
sd_plot --cbtiks 5 
cp *.png ../mag_cplx4.png
rm *.png
sd_plot -v 1.5 -V 2.5
cp *.png ../mag_cplx5.png
rm *.png

sd_plot -t pha
cp *.png ../pha_cplx1.png
rm *.png
sd_plot -t pha -x -10 -X 40 -z -10 -Z 0 
cp *.png ../pha_cplx2.png
rm *.png
sd_plot -t pha -u cm --alpha_cov --cmaglin 
cp *.png ../pha_cplx3.png
rm *.png
sd_plot -t pha --cbtiks 5 
cp *.png ../pha_cplx4.png
rm *.png
sd_plot -t pha -v -1.5 -V 0
cp *.png ../pha_cplx5.png
rm *.png

# test plots for FPI data in sipdir
cd ../sipdir_fpi
sd_plot
cp *.png ../mag_fpi1.png
rm *.png
sd_plot -x -10 -X 40 -z -10 -Z 0 
cp *.png ../mag_fpi2.png
rm *.png
sd_plot -u cm --alpha_cov --cmaglin 
cp *.png ../mag_fpi3.png
rm *.png
sd_plot --cbtiks 5
cp *.png ../mag_fpi4.png
rm *.png
sd_plot -v 1.5 -V 2.5
cp *.png ../mag_fpi5.png
rm *.png

sd_plot -t pha
cp *.png ../pha_fpi1.png
rm *.png
sd_plot -t pha -x -10 -X 40 -z -10 -Z 0 
cp *.png ../pha_fpi2.png
rm *.png
sd_plot -t pha -u cm --alpha_cov --cmaglin 
cp *.png ../pha_fpi3.png
rm *.png
sd_plot -t pha --cbtiks 5
cp *.png ../pha_fpi4.png
rm *.png
sd_plot -t pha -v -1.5 -V 0
cp *.png ../pha_fpi5.png
rm *.png

# test plots for DC data in sipdir
cd ../sipdir_mag
sd_plot 
cp *.png ../mag_dc1.png
rm *.png
sd_plot -x -10 -X 40 -z -10 -Z 0 
cp *.png ../mag_dc2.png
rm *.png
sd_plot -u cm --alpha_cov --cmaglin
cp *.png ../mag_dc3.png
rm *.png
sd_plot --cbtiks 5 
cp *.png ../mag_dc4.png
rm *.png
sd_plot -v 1.5 -V 2.5
cp *.png ../mag_dc5.png
rm *.png
