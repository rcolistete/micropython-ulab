
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

#define ITERATE_VECTOR(type, array, out) do {\
    type *input = (type *)(array);\
    for(size_t i=0; i < (source)->len; i++) {\
		*out++ = f(*input++);\
    }\
} while(0)

#define ITERATE_VECTOR_SLICE(type, source, out) do{\
    type *input = (type *)(source)->array;\
	size_t coords[ULAB_MAX_DIMS];\
	for(uint8_t i=0; i < ULAB_MAX_DIMS; i++) coords[i] = 0;\
    size_t offset = 0;\
	for(size_t i=0; i < (source)->len; i++) {\
		mp_float_t value = ndarray_get_float_value(input, (source)->dtype, offset);\
		(out)[i] = f(value);\
		offset += (source)->strides[(source)->ndim-1];\
		coords[(source)->ndim-1] += 1;\
		for(uint8_t j=(source)->ndim-1; j > 0; j--) {\
			if(coords[j] == (source)->shape[j]) {\
				offset -= (source)->shape[j] * (source)->strides[j];\
				offset += (source)->strides[j-1];\
				coords[j] = 0;\
				coords[j-1] += 1;\
			} else {\
				break;\
			}\
		}\
    }\
} while(0)

#define MATH_FUN_1(py_name, c_name) \
    mp_obj_t vectorise_ ## py_name(mp_obj_t x_obj) { \
        return vectorise_generic_vector(x_obj, MICROPY_FLOAT_C_FUN(c_name)); \
}
    
#endif
#endif
