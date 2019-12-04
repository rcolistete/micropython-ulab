/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 Zoltán Vörös
*/
    
#ifndef _NUMERICAL_
#define _NUMERICAL_

#include "ndarray.h"

mp_obj_t numerical_linspace(size_t , const mp_obj_t *, mp_map_t *);

mp_obj_t numerical_sum(size_t , const mp_obj_t *, mp_map_t *);
mp_obj_t numerical_mean(size_t , const mp_obj_t *, mp_map_t *);
mp_obj_t numerical_std(size_t , const mp_obj_t *, mp_map_t *);

#define CALCULATE_SUM(ndarray, type, farray, shape_ax, index, stride, offset, optype) do {\
    type *array = (type *)(ndarray)->array->items;\
    (farray)[(index)] = 0.0;\
    for(size_t j=0; j < (shape_ax); j++, (offset) += (stride)) {\
        (farray)[(index)] += array[(offset)];\
    }\
} while(0)

// TODO: this can be done without the NDARRAY_INDEX_FROM_FLAT macro
// Welford algorithm for the standard deviation
#define CALCULATE_FLAT_SUM_STD(ndarray, type, value, shape_strides, len, optype) do {\
    type *array = (type *)(ndarray)->array->items;\
    (value) = 0.0;\
    mp_float_t m = 0.0, mtmp;\
    size_t index, nindex;\
    for(size_t j=0; j < (len); j++) {\
        NDARRAY_INDEX_FROM_FLAT((ndarray), (shape_strides), j, index, nindex);\
        if((optype) == NUMERICAL_STD) {\
            mtmp = m;\
            m = mtmp + (array[nindex] - mtmp) / (j+1);\
            (value) += (array[nindex] - mtmp) * (array[nindex] - m);\
        } else {\
            (value) += array[nindex];\
        }\
    }\
} while(0)

// we calculate the standard deviation in two passes, in order to avoid negative values through truncation errors
// We could do in a single pass, if we resorted to the Welford algorithm above
#define CALCULATE_STD(ndarray, type, sq_sum, shape_ax, stride, offset) do {\
    type *array = (type *)(ndarray)->array->items;\
    mp_float_t x, ave = 0.0;\
    (sq_sum) = 0.0;\
    size_t j, _offset = (offset);\
    for(j=0; j < (shape_ax); j++, _offset += (stride)) {\
        ave += array[_offset];\
    }\
    ave /= j;\
    for(j=0; j < (shape_ax); j++, (offset) += (stride)) {\
        x = array[(offset)] - ave;\
        sq_sum += x * x;\
    }\
} while(0)

#endif
