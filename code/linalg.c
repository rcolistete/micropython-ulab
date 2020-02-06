/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019-2020 Zoltán Vörös
*/
    
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "py/obj.h"
#include "py/runtime.h"
#include "py/misc.h"
#include "linalg.h"

#if ULAB_LINALG_ZEROS || ULAB_LINALG_ONES
mp_obj_t linalg_zeros_ones(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args, uint8_t kind) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 1} } ,
        { MP_QSTR_dtype, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = NDARRAY_FLOAT} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    
    ndarray_obj_t *ndarray = ndarray_new_ndarray(args[0].u_int, args[1].u_int);

    if(kind == 1) {
        mp_obj_t one = mp_obj_new_int(1);
        for(size_t i=0; i < ndarray->array->len; i++) {
            mp_binary_set_val_array(args[1].u_int, ndarray->array->items, i, one);
        }
    }
    return MP_OBJ_FROM_PTR(ndarray);
}
#endif

#if ULAB_LINALG_ZEROS
mp_obj_t linalg_zeros(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return linalg_zeros_ones(n_args, pos_args, kw_args, 0);
}

MP_DEFINE_CONST_FUN_OBJ_KW(linalg_zeros_obj, 0, linalg_zeros);
#endif

#if ULAB_LINALG_ONES
mp_obj_t linalg_ones(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return linalg_zeros_ones(n_args, pos_args, kw_args, 1);
}

MP_DEFINE_CONST_FUN_OBJ_KW(linalg_ones_obj, 0, linalg_ones);
#endif
