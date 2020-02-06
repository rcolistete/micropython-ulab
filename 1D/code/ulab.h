/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019-2020 Zoltán Vörös
*/
    
#ifndef __ULAB__

#define __ULAB__

// vectorise takes approx. 3 kB of flash space
#define ULAB_VECTORISE_ACOS (0)
#define ULAB_VECTORISE_ACOSH (0)
#define ULAB_VECTORISE_ASIN (0)
#define ULAB_VECTORISE_ASINH (0)
#define ULAB_VECTORISE_ATAN (0)
#define ULAB_VECTORISE_ATANH (0)
#define ULAB_VECTORISE_CEIL (0)
#define ULAB_VECTORISE_COS (0)
#define ULAB_VECTORISE_ERF (0)
#define ULAB_VECTORISE_ERFC (0)
#define ULAB_VECTORISE_EXP (0)
#define ULAB_VECTORISE_EXPM1 (0)
#define ULAB_VECTORISE_FLOOR (0)
#define ULAB_VECTORISE_GAMMA (0)
#define ULAB_VECTORISE_LGAMMA (0)
#define ULAB_VECTORISE_LOG (0)
#define ULAB_VECTORISE_LOG10 (0)
#define ULAB_VECTORISE_LOG2 (0)
#define ULAB_VECTORISE_SIN (0)
#define ULAB_VECTORISE_SINH (0)
#define ULAB_VECTORISE_SQRT (0)
#define ULAB_VECTORISE_TAN (0)
#define ULAB_VECTORISE_TANH (0)

// linalg adds around 450 bytes
#define ULAB_LINALG_ZEROS (0)
#define ULAB_LINALG_ONES (0)

// poly is approx. 2.5 kB
#define ULAB_POLY_POLYVAL (0)
#define ULAB_POLY_POLYFIT (0)

#define ULAB_NUMERICAL_LINSPACE (1)
#define ULAB_NUMERICAL_SUM (1)
#define ULAB_NUMERICAL_MEAN (1)
#define ULAB_NUMERICAL_STD (1)

// FFT costs about 2 kB of flash space
#define ULAB_FFT_FFT (0)
#define ULAB_FFT_IFFT (0)
#define ULAB_FFT_SPECTRUM (0)

#endif
