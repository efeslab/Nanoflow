Command
```bash
arg1: verification (0=no, 1=yes)
arg2: initialization (0=no init, 1=integer value, 2=decimal value)
arg3: time kernel (0=no, 1=yes)
Following arguments (depending on number of spatial dims):
 Number of spatial dimensions (1=Conv1D, 2=Conv2D, 3=Conv3D)
 G, N, K, C,
 <filter spatial dimensions>, (ie Y, X for 2D)
 <input image spatial dimensions>, (ie Hi, Wi for 2D)
 <strides>, (ie Sy, Sx for 2D)
 <dilations>, (ie Dy, Dx for 2D)
 <left padding>, (ie LeftPy, LeftPx for 2D)
 <right padding>, (ie RightPy, RightPx for 2D)

./bin/example_grouped_conv_fwd_bias_relu_add_xdl_fp16 1 1 1
```

