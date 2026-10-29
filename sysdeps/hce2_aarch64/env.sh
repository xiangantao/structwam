# Self-contained system libraries for LIBERO-plus on Huawei Cloud EulerOS 2.0 aarch64 images that lack
# ImageMagick (python wand, sensor-noise perturbations) and mesa-dri-drivers / llvm-libs (MuJoCo EGL
# software rendering).  Copied from notebook 30232 (ImageMagick-7.1.2.29, mesa-dri-drivers-21.3.1,
# llvm-libs-12.0.1, plus the libraries those need that the stock image does not ship).
#   source /home/ma-user/work/2337888765/sysdeps/hce2_aarch64/env.sh
SYSDEPS=/home/ma-user/work/2337888765/sysdeps/hce2_aarch64
export LD_LIBRARY_PATH="$SYSDEPS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAGICK_HOME="$SYSDEPS"
export MAGICK_CODER_MODULE_PATH="$SYSDEPS/lib/ImageMagick-7.1.2/modules-Q16HDRI/coders"
export MAGICK_CODER_FILTER_PATH="$SYSDEPS/lib/ImageMagick-7.1.2/modules-Q16HDRI/filters"
export MAGICK_CONFIGURE_PATH="$SYSDEPS/etc/ImageMagick-7:$SYSDEPS/lib/ImageMagick-7.1.2/config-Q16HDRI"
export LIBGL_DRIVERS_PATH="$SYSDEPS/dri"
