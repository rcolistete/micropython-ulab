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

#if ULAB_NUMERICAL_LINSPACE
mp_obj_t numerical_linspace(size_t , const mp_obj_t *, mp_map_t *);
MP_DECLARE_CONST_FUN_OBJ_KW(numerical_linspace_obj);
#endif

#if ULAB_NUMERICAL_SUM
mp_obj_t numerical_sum(size_t , const mp_obj_t *, mp_map_t *);
MP_DECLARE_CONST_FUN_OBJ_KW(numerical_sum_obj);
#endif

#if ULAB_NUMERICAL_MEAN
mp_obj_t numerical_mean(size_t , const mp_obj_t *, mp_map_t *);
MP_DECLARE_CONST_FUN_OBJ_KW(numerical_mean_obj);
#endif

#if ULAB_NUMERICAL_STD
mp_obj_t numerical_std(size_t , const mp_obj_t *, mp_map_t *);
MP_DECLARE_CONST_FUN_OBJ_KW(numerical_std_obj);
#endif

#define CALCULATE_SUM_STD_1D(ndarray, type, value, offset, optype) do {\
    type *array = (type *)(ndarray)->array->items;\
    type tmp;\
    (value) = 0.0;\
    mp_float_t m = 0.0, mtmp;\
    for(size_t j=0; j < (ndarray)->len; j++) {\
        tmp = array[(offset)];\
        if((optype) == NUMERICAL_STD) {\
            mtmp = m;\
            m = mtmp + (tmp - mtmp) / (j+1);\
            (value) += (tmp - mtmp) * (tmp - m);\
        } else {\
            (value) += tmp;\
        }\
        (offset) += (ndarray)->stride;\
    }\
} while(0)

#endif
