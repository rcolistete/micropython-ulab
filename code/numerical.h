
/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019-2020 Zoltán Vörös
*/

#ifndef _NUMERICAL_
#define _NUMERICAL_

#include "ulab.h"
#include "ndarray.h"

#if ULAB_NUMERICAL_MODULE

mp_obj_module_t ulab_numerical_module;

// TODO: implement minimum/maximum, and cumsum
//mp_obj_t numerical_minimum(mp_obj_t , mp_obj_t );
//mp_obj_t numerical_maximum(mp_obj_t , mp_obj_t );
//mp_obj_t numerical_cumsum(size_t , const mp_obj_t *, mp_map_t *);

#define CALCULATE_FLAT_SUM_STD(ndarray, coords, type, value, optype) do {\
    type *array = (type *)(ndarray)->array;\
    (value) = 0.0;\
    int32_t offset = 0;\
    mp_float_t m = 0.0, mtmp;\
	for(size_t i=0; i < (ndarray)->len; i++) {\
		offset += (ndarray)->strides[(ndarray)->ndim-1];\
		(coords)[(ndarray)->ndim-1] += 1;\
		if((optype) == NUMERICAL_STD) {\
			mtmp = m;\
			m = mtmp + (array[offset] - mtmp) / (i+1);\
			(value) += (array[offset] - mtmp) * (array[offset] - m);\
		} else {\
			(value) += array[offset];\
		}\
		for(uint8_t j=(ndarray)->ndim-1; j > 0; j--) {\
			if((coords)[j] == (ndarray)->shape[j]) {\
				offset -= (ndarray)->shape[j] * (ndarray)->strides[j];\
				offset += (ndarray)->strides[j-1];\
				(coords)[j] = 0;\
				(coords)[j-1] += 1;\
			} else {\
				break;\
			}\
		}\
	}\
} while(0)

#define RUN_ARGMIN(in, out, typein, typeout, len, start, increment, op, pos) do {\
    typein *array = (typein *)(in)->array->items;\
    typeout *outarray = (typeout *)(out)->array->items;\
    size_t best_index = 0;\
    if(((op) == NUMERICAL_MAX) || ((op) == NUMERICAL_ARGMAX)) {\
        for(size_t i=1; i < (len); i++) {\
            if(array[(start)+i*(increment)] > array[(start)+best_index*(increment)]) best_index = i;\
        }\
        if((op) == NUMERICAL_MAX) outarray[(pos)] = array[(start)+best_index*(increment)];\
        else outarray[(pos)] = best_index;\
    } else{\
        for(size_t i=1; i < (len); i++) {\
            if(array[(start)+i*(increment)] < array[(start)+best_index*(increment)]) best_index = i;\
        }\
        if((op) == NUMERICAL_MIN) outarray[(pos)] = array[(start)+best_index*(increment)];\
        else outarray[(pos)] = best_index;\
    }\
} while(0)

#define RUN_SUM(ndarray, type, optype, len, start, increment) do {\
    type *array = (type *)(ndarray)->array->items;\
    type value;\
    for(size_t j=0; j < (len); j++) {\
        value = array[(start)+j*(increment)];\
        sum += value;\
    }\
} while(0)

#define RUN_STD(ndarray, type, len, start, increment) do {\
    type *array = (type *)(ndarray)->array->items;\
    mp_float_t value;\
    for(size_t j=0; j < (len); j++) {\
        sum += array[(start)+j*(increment)];\
    }\
    sum /= (len);\
    for(size_t j=0; j < (len); j++) {\
        value = (array[(start)+j*(increment)] - sum);\
        sum_sq += value * value;\
    }\
} while(0)

#define CALCULATE_DIFF(in, out, type, axis, coords, shape, strides) do {\
    type *source = (type *)(in)->array;\
    type *target = (type *)(out)->array;\
    size_t reduced_size = (in)->len/(in)->shape[(axis)];\
	size_t offset = 0;\
	if((in)->ndim == 1) {\
		for(size_t j=0; j < (ndarray)->shape[0]-1; i++) {\
			target[j] = source[offset+(ndarray)->strides[0]] - source[offset];\
			offset += (ndarray)->strides[0];\
		}\
	} else {\
		for(size_t i=0; i < reduced_size; i++) {\
			for(size_t j=0; j < (ndarray)->shape[(axis)]-1; i++) {\
				target[offset+j] = source[offset+j+(ndarray)->strides[(axis)]] - source[offset+j];\
			}\
			offset += (shape)[(ndarray)->ndim-2];\
			(coords)[(ndarray)->ndim-2] += 1;\
			for(uint8_t k=(ndim)->ndim-2; k > 0; k--) {\
				if((coords)[k] == (shape)[k]) {\
					offset -= (shape)[k] * (strides)[k];\
					offset += (strides)[k-1];\
					(coords)[k] = 0;\
					(coords)[k-1] += 1;\
				} else {\
					break;\
				}\
			}\
		}\
	}\
} while(0)

#define HEAPSORT(type, ndarray) do {\
    type *array = (type *)(ndarray)->array;\
    type tmp;\
    for (;;) {\
        if (k > 0) {\
            tmp = array[start+(--k)*increment];\
        } else {\
            q--;\
            if(q == 0) {\
                break;\
            }\
            tmp = array[start+q*increment];\
            array[start+q*increment] = array[start];\
        }\
        p = k;\
        c = k + k + 1;\
        while (c < q) {\
            if((c + 1 < q)  &&  (array[start+(c+1)*increment] > array[start+c*increment])) {\
                c++;\
            }\
            if(array[start+c*increment] > tmp) {\
                array[start+p*increment] = array[start+c*increment];\
                p = c;\
                c = p + p + 1;\
            } else {\
                break;\
            }\
        }\
        array[start+p*increment] = tmp;\
    }\
} while(0)

// This is pretty similar to HEAPSORT above; perhaps, the two could be combined somehow
// On the other hand, since this is a macro, it doesn't really matter
// Keep in mind that initially, index_array[start+s*increment] = s
#define HEAP_ARGSORT(type, ndarray, index_array) do {\
    type *array = (type *)(ndarray)->array;\
    type tmp;\
    uint16_t itmp;\
    for (;;) {\
        if (k > 0) {\
            k--;\
            tmp = array[start+index_array[start+k*increment]*increment];\
            itmp = index_array[start+k*increment];\
        } else {\
            q--;\
            if(q == 0) {\
                break;\
            }\
            tmp = array[start+index_array[start+q*increment]*increment];\
            itmp = index_array[start+q*increment];\
            index_array[start+q*increment] = index_array[start];\
        }\
        p = k;\
        c = k + k + 1;\
        while (c < q) {\
            if((c + 1 < q)  &&  (array[start+index_array[start+(c+1)*increment]*increment] > array[start+index_array[start+c*increment]*increment])) {\
                c++;\
            }\
            if(array[start+index_array[start+c*increment]*increment] > tmp) {\
                index_array[start+p*increment] = index_array[start+c*increment];\
                p = c;\
                c = p + p + 1;\
            } else {\
                break;\
            }\
        }\
        index_array[start+p*increment] = itmp;\
    }\
} while(0)

#endif
#endif
