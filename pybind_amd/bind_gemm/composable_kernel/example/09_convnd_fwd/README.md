# Instructions for ```example_convnd_fwd_xdl```

## Run ```example_convnd_fwd_xdl```
```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
#arg4: N spatial dimensions (default 2)
#Following arguments (depending on number of spatial dims):
# N, K, C, 
# <filter spatial dimensions>, (ie Y, X for 2D)
# <input image spatial dimensions>, (ie Hi, Wi for 2D)
# <strides>, (ie Sy, Sx for 2D)
# <dilations>, (ie Dy, Dx for 2D)
# <left padding>, (ie LeftPy, LeftPx for 2D)
# <right padding>, (ie RightPy, RightPx for 2D)
./bin/example_convnd_fwd_xdl 0 1 100
```
