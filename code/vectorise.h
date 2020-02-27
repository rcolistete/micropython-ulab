
/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019-2020 Zoltán Vörös
*/

#ifndef _VECTORISE_
#define _VECTORISE_

#include "ulab.h"
#include "ndarray.h"

#if ULAB_VECTORISE_MODULE

mp_obj_module_t ulab_vectorise_module;

#define ITERATE_VECTOR(type, source, out) do {\
    type *input = (type *)(source)->array;\
    for(size_t i=0; i < (source)->len; i++) {\
                *out++ = f(*input++);\
    }\
} while(0)

#define ITERATE_VECTOR_SLICE(type, source, out, strides_array, shape_strides) do{\
    size_t tindex, nindex;\
    type *input = (type *)(source)->array;\
    for(size_t i=0; i < len; i++) {\
        NDARRAY_INDEX_FROM_FLAT2((source), (strides_array), (shape_strides), i, tindex, nindex);\
        (out)[i] = f(input[nindex]);\
    }\
} while(0)
    
#define ITERATE_VECTOR_SLICE(type, source, out, strides_array, shape_strides) do{\
    size_t tindex, nindex;\
    type *input = (type *)(source)->array;\
    for(size_t i=0; i < len; i++) {\
        NDARRAY_INDEX_FROM_FLAT2((source), (strides_array), (shape_strides), i, tindex, nindex);\
        (out)[i] = f(input[nindex]);\
    }\
} while(0)

#define MATH_FUN_1(py_name, c_name) \
    mp_obj_t vectorise_ ## py_name(mp_obj_t x_obj) { \
        return vectorise_generic_vector(x_obj, MICROPY_FLOAT_C_FUN(c_name)); \
}
    
#endif
#endif
