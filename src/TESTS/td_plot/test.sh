#!/bin/bash

# test plots for complex data in tomodir
cd tomodir_complex
td_plot --title 'no options'
cp *.png ../overview_cplx1.png
rm *.png
td_plot -x 0 -X 45 -z 125 -Z 145 --title 'axis limits'
cp *.png ../overview_cplx2.png
rm *.png
td_plot -u cm --alpha_cov --cmaglin --title 'unit, transparency, linear magnitude'
cp *.png ../overview_cplx3.png
rm *.png
td_plot --cov_cbtiks 5 --mag_cbtiks 5 --pha_cbtiks 5 --real_cbtiks 5 --imag_cbtiks 5 --title 'CB tiks'
cp *.png ../overview_cplx4.png
rm *.png
td_plot --no_elecs --title 'no electrodes'
cp *.png ../overview_cplx5.png
rm *.png
td_plot --cov_vmin -3.5 --mag_vmin .5 --pha_vmin -.1 --real_vmin -2.5 --imag_vmin -6.4 --cov_vmax 0 --mag_vmax 2.5 --pha_vmax -.05 --real_vmax -.5 --imag_vmax -4.5 --title 'CB min max'
cp *.png ../overview_cplx6.png
rm *.png
td_plot --single
cp c*.png ../singlecov_cplx1.png
cp rh*.png ../singlerho_cplx1.png
cp re*.png ../singlereal_cplx1.png
cp p*.png ../singlephi_cplx1.png
cp i*.png ../singleimag_cplx1.png
rm *.png
td_plot --single -x 0 -X 45 -z 125 -Z 145 --title 'axis limits'
cp c*.png ../singlecov_cplx2.png
cp rh*.png ../singlerho_cplx2.png
cp re*.png ../singlereal_cplx2.png
cp p*.png ../singlephi_cplx2.png
cp i*.png ../singleimag_cplx2.png
td_plot --single -u cm --alpha_cov --cmaglin --title 'unit, transparency, linear magnitude'
cp c*.png ../singlecov_cplx3.png
cp rh*.png ../singlerho_cplx3.png
cp re*.png ../singlereal_cplx3.png
cp p*.png ../singlephi_cplx3.png
cp i*.png ../singleimag_cplx3.png
rm *.png
td_plot --single --mag_cbtiks 5 --imag_cbtiks 5 --real_cbtiks 5 --pha_cbtiks 5 --cov_cbtiks 5 --title 'CB tiks'
cp c*.png ../singlecov_cplx4.png
cp rh*.png ../singlerho_cplx4.png
cp re*.png ../singlereal_cplx4.png
cp p*.png ../singlephi_cplx4.png
cp i*.png ../singleimag_cplx4.png
rm *.png
td_plot --single --no_elecs --title 'no electrodes'
cp c*.png ../singlecov_cplx5.png
cp rh*.png ../singlerho_cplx5.png
cp re*.png ../singlereal_cplx5.png
cp p*.png ../singlephi_cplx5.png
cp i*.png ../singleimag_cplx5.png
rm *.png
td_plot --single --mag_vmin .5 --mag_vmax 2.5 --pha_vmin -3.5 --pha_vmax 0 --imag_vmin .5 --imag_vmax 2.5 --real_vmin .5 --real_vmax 2.5 --cov_vmin -2.5 --cov_vmax -.5  --title 'CB min max'
cp c*.png ../singlecov_cplx6.png
cp rh*.png ../singlerho_cplx6.png
cp re*.png ../singlereal_cplx6.png
cp p*.png ../singlephi_cplx6.png
cp i*.png ../singleimag_cplx6.png
rm *.png

# test plots for FPI data in tomodir
cd ../tomodir_fpi
td_plot --title 'no options'
cp *.png ../overview_fpi1.png
rm *.png
td_plot -x 0 -X 45 -z 125 -Z 145 --title 'axis limits'
cp *.png ../overview_fpi2.png
rm *.png
td_plot -u cm --alpha_cov --cmaglin --title 'unit, transparency, linear magnitude'
cp *.png ../overview_fpi3.png
rm *.png
td_plot --cov_cbtiks 5 --mag_cbtiks 5 --pha_cbtiks 5 --real_cbtiks 5 --imag_cbtiks 5 --title 'CB tiks'
cp *.png ../overview_fpi4.png
rm *.png
td_plot --no_elecs --title 'no electrodes'
cp *.png ../overview_fpi5.png
rm *.png
td_plot --cov_vmin -3.5 --mag_vmin .5 --pha_vmin -.1 --real_vmin -2.5 --imag_vmin -6.4 --cov_vmax 0 --mag_vmax 2.5 --pha_vmax -.05 --real_vmax -.5 --imag_vmax -4.5 --title 'CB min max'
cp *.png ../overview_fpi6.png
rm *.png
td_plot --single
cp c*.png ../singlecov_fpi1.png
cp rh*.png ../singlerho_fpi1.png
cp re*.png ../singlereal_fpi1.png
cp p*.png ../singlephi_fpi1.png
cp i*.png ../singleimag_fpi1.png
cp fpi_i*.png ../singlefimag_fpi1.png
cp fpi_r*.png ../singlefreal_fpi1.png
cp fpi_p*.png ../singlefphi_fpi1.png
rm *.png
td_plot --single -x 0 -X 45 -z 125 -Z 145 --title 'axis limits'
cp c*.png ../singlecov_fpi2.png
cp rh*.png ../singlerho_fpi2.png
cp re*.png ../singlereal_fpi2.png
cp p*.png ../singlephi_fpi2.png
cp i*.png ../singleimag_fpi2.png
cp fpi_i*.png ../singlefimag_fpi2.png
cp fpi_r*.png ../singlefreal_fpi2.png
cp fpi_p*.png ../singlefphi_fpi2.png
rm *.png
td_plot --single -u cm --alpha_cov --cmaglin --title 'unit, transparency, linear magnitude'
cp c*.png ../singlecov_fpi3.png
cp rh*.png ../singlerho_fpi3.png
cp re*.png ../singlereal_fpi3.png
cp p*.png ../singlephi_fpi3.png
cp i*.png ../singleimag_fpi3.png
cp fpi_i*.png ../singlefimag_fpi3.png
cp fpi_r*.png ../singlefreal_fpi3.png
cp fpi_p*.png ../singlefphi_fpi3.png
rm *.png
td_plot --single --mag_cbtiks 5 --imag_cbtiks 5 --real_cbtiks 5 --pha_cbtiks 5 --cov_cbtiks 5 --title 'CB tiks'
cp c*.png ../singlecov_fpi4.png
cp rh*.png ../singlerho_fpi4.png
cp re*.png ../singlereal_fpi4.png
cp p*.png ../singlephi_fpi4.png
cp i*.png ../singleimag_fpi4.png
cp fpi_i*.png ../singlefimag_fpi4.png
cp fpi_r*.png ../singlefreal_fpi4.png
cp fpi_p*.png ../singlefphi_fpi4.png
rm *.png
td_plot --single --no_elecs --title 'no electrodes'
cp c*.png ../singlecov_fpi5.png
cp rh*.png ../singlerho_fpi5.png
cp re*.png ../singlereal_fpi5.png
cp p*.png ../singlephi_fpi5.png
cp i*.png ../singleimag_fpi5.png
cp fpi_i*.png ../singlefimag_fpi5.png
cp fpi_r*.png ../singlefreal_fpi5.png
cp fpi_p*.png ../singlefphi_fpi5.png
rm *.png
td_plot --single --mag_vmin .5 --mag_vmax 2.5 --pha_vmin -3.5 --pha_vmax 0 --imag_vmin .5 --imag_vmax 2.5 --real_vmin .5 --real_vmax 2.5 --cov_vmin -2.5 --cov_vmax -.5  --title 'CB min max'
cp c*.png ../singlecov_fpi6.png
cp rh*.png ../singlerho_fpi6.png
cp re*.png ../singlereal_fpi6.png
cp p*.png ../singlephi_fpi6.png
cp i*.png ../singleimag_fpi6.png
cp fpi_i*.png ../singlefimag_fpi6.png
cp fpi_r*.png ../singlefreal_fpi6.png
cp fpi_p*.png ../singlefphi_fpi6.png
rm *.png

# test plots for DC data in tomodir
cd ../tomodir_mag
td_plot --title 'no options'
cp *.png ../overview_mag1.png
rm *.png
td_plot -x 0 -X 45 -z 125 -Z 145 --title 'axis limits'
cp *.png ../overview_mag2.png
rm *.png
td_plot -u cm --alpha_cov --cmaglin --title 'unit, transparency, linear magnitude'
cp *.png ../overview_mag3.png
rm *.png
td_plot --cov_cbtiks 5 --mag_cbtiks 5 --pha_cbtiks 5 --real_cbtiks 5 --imag_cbtiks 5 --title 'CB tiks'
cp *.png ../overview_mag4.png
rm *.png
td_plot --no_elecs --title 'no electrodes'
cp *.png ../overview_mag5.png
rm *.png
td_plot --cov_vmin -3.5 --mag_vmin .5 --cov_vmax 0 --mag_vmax 2.5  --title 'CB min max'
cp *.png ../overview_mag6.png
rm *.png
td_plot --single 
cp c*.png ../singlecov_dc1.png
cp rh*.png ../singlerho_dc1.png
rm *.png
td_plot --single -x 0 -X 45 -z 125 -Z 145 --title 'axis limits'
cp c*.png ../singlecov_dc2.png
cp rh*.png ../singlerho_dc2.png
rm *.png
td_plot --single -u cm --alpha_cov --cmaglin --title 'unit, transparency, linear magnitude'
cp c*.png ../singlecov_dc3.png
cp rh*.png ../singlerho_dc3.png
rm *.png
td_plot --single --mag_cbtiks 5 --cov_cbtiks 5 --title 'CB tiks'
cp c*.png ../singlecov_dc4.png
cp rh*.png ../singlerho_dc4.png
rm *.png
td_plot --single --no_elecs --title 'no electrodes'
cp c*.png ../singlecov_dc5.png
cp rh*.png ../singlerho_dc5.png
rm *.png
td_plot --single --mag_vmin .5 --mag_vmax 2.5 --cov_vmin -2.5 --cov_vmax -.5  --title 'CB min max'
cp c*.png ../singlecov_dc6.png
cp rh*.png ../singlerho_dc6.png
rm *.png

# test plots for timelapse data in tomodir
cd ../tomodir_timelapse
td_plot -c 3 --title 'column 4'
cp *.png ../overview_tl1.png
rm *.png
td_plot --single -c 3 --title 'column 4'
cp c*.png ../singlecov_tl1.png
cp rh*.png ../singlerho_tl1.png
rm *.png
td_plot --single -c 4 --title 'column 5'
cp c*.png ../singlecov_tl2.png
cp rh*.png ../singlerho_tl2.png
rm *.png
td_plot --single -c 5 --title 'column 6'
cp c*.png ../singlecov_tl3.png
cp rh*.png ../singlerho_tl3.png
rm *.png
td_plot --single -c 6 --title 'column 7'
cp c*.png ../singlecov_tl4.png
cp rh*.png ../singlerho_tl4.png
rm *.png
td_plot --single -c 7 --title 'non exsting column'

# test plots for anisotropic data in tomodir
cd ../tomodir_aniso
td_plot --aniso
cp m*.png ../aniso_mag1.png
cp p*.png ../aniso_pha1.png
rm *.png
