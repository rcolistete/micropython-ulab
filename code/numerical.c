/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019-2020 Zoltán Vörös
*/
    
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "py/obj.h"
#include "py/objint.h"
#include "py/runtime.h"
#include "py/builtin.h"
#include "py/misc.h"
#include "numerical.h"

enum NUMERICAL_FUNCTION_TYPE {
    NUMERICAL_MIN,
    NUMERICAL_MAX,
    NUMERICAL_ARGMIN,
    NUMERICAL_ARGMAX,
    NUMERICAL_SUM,
    NUMERICAL_MEAN,
    NUMERICAL_STD,
};

mp_obj_t numerical_linspace(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_PTR(&mp_const_none_obj) } },
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_PTR(&mp_const_none_obj) } },
        { MP_QSTR_num, MP_ARG_INT, {.u_int = 50} },
        { MP_QSTR_endpoint, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_PTR(&mp_const_true_obj)} },
        { MP_QSTR_retstep, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_PTR(&mp_const_false_obj)} },
        { MP_QSTR_dtype, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = NDARRAY_FLOAT} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(2, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    uint16_t len = args[2].u_int;
    if(len < 2) {
        mp_raise_ValueError(translate("number of points must be at least 2"));
    }
    mp_float_t value, step;
    value = mp_obj_get_float(args[0].u_obj);
    uint8_t typecode = args[5].u_int;
    if(args[3].u_obj == mp_const_true) step = (mp_obj_get_float(args[1].u_obj)-value)/(len-1);
    else step = (mp_obj_get_float(args[1].u_obj)-value)/len;
    ndarray_obj_t *ndarray = ndarray_new_ndarray(len, typecode);
    if(typecode == NDARRAY_UINT8) {
        uint8_t *array = (uint8_t *)ndarray->array->items;
        for(size_t i=0; i < len; i++, value += step) array[i] = (uint8_t)value;
    } else if(typecode == NDARRAY_INT8) {
        int8_t *array = (int8_t *)ndarray->array->items;
        for(size_t i=0; i < len; i++, value += step) array[i] = (int8_t)value;
    } else if(typecode == NDARRAY_UINT16) {
        uint16_t *array = (uint16_t *)ndarray->array->items;
        for(size_t i=0; i < len; i++, value += step) array[i] = (uint16_t)value;
    } else if(typecode == NDARRAY_INT16) {
        int16_t *array = (int16_t *)ndarray->array->items;
        for(size_t i=0; i < len; i++, value += step) array[i] = (int16_t)value;
    } else {
        mp_float_t *array = (mp_float_t *)ndarray->array->items;
        for(size_t i=0; i < len; i++, value += step) array[i] = value;
    }
    if(args[4].u_obj == mp_const_false) {
        return MP_OBJ_FROM_PTR(ndarray);
    } else {
        mp_obj_t tuple[2];
        tuple[0] = ndarray;
        tuple[1] = mp_obj_new_float(step);
        return mp_obj_new_tuple(2, tuple);
    }
}

// numerical functions for ndarrays
mp_obj_t numerical_sum_mean_std_ndarray(ndarray_obj_t *ndarray, uint8_t optype, size_t ddof) {
    size_t offset = ndarray->offset;
    mp_float_t value;
    if(ndarray->array->typecode == NDARRAY_UINT8) {
        CALCULATE_SUM_STD_1D(ndarray, uint8_t, value, offset, optype);
    } else if(ndarray->array->typecode == NDARRAY_INT8) {
        CALCULATE_SUM_STD_1D(ndarray, int8_t, value, offset, optype);
    } else if(ndarray->array->typecode == NDARRAY_UINT16) {
        CALCULATE_SUM_STD_1D(ndarray, uint16_t, value, offset, optype);
    } else if(ndarray->array->typecode == NDARRAY_INT16) {
        CALCULATE_SUM_STD_1D(ndarray, int16_t, value, offset, optype);
    } else {
        CALCULATE_SUM_STD_1D(ndarray, mp_float_t, value, offset, optype);
    }
    if(optype == NUMERICAL_MEAN) {
        value /= ndarray->len;
        return mp_obj_new_float(value);
    } else if(optype == NUMERICAL_STD) {
        value = MICROPY_FLOAT_C_FUN(sqrt)(value/(ndarray->len-ddof));
    } 
    if(ndarray->array->typecode == NDARRAY_FLOAT) {
        return mp_obj_new_float(value);
    }
    return mp_obj_new_int(value);
}

mp_obj_t numerical_argmin_argmax_ndarray(ndarray_obj_t *ndarray, uint8_t optype) {
    return mp_const_none;
}

// numerical function for interables (single axis)
mp_obj_t numerical_argmin_argmax_iterable(mp_obj_t oin, uint8_t optype) {
    size_t idx = 0, best_idx = 0;
    mp_obj_iter_buf_t iter_buf;
    mp_obj_t iterable = mp_getiter(oin, &iter_buf);
    mp_obj_t best_obj = MP_OBJ_NULL;
    mp_obj_t item;
    mp_uint_t op = MP_BINARY_OP_LESS;
    if((optype == NUMERICAL_ARGMAX) || (optype == NUMERICAL_MAX)) op = MP_BINARY_OP_MORE;
    while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
        if ((best_obj == MP_OBJ_NULL) || (mp_binary_op(op, item, best_obj) == mp_const_true)) {
            best_obj = item;
            best_idx = idx;
        }
        idx++;
    }
    if((optype == NUMERICAL_ARGMIN) || (optype == NUMERICAL_ARGMAX)) {
        return MP_OBJ_NEW_SMALL_INT(best_idx);
    } else {
        return best_obj;
    }    
}

mp_obj_t numerical_sum_mean_std_iterable(mp_obj_t oin, uint8_t optype, size_t ddof) {
    mp_float_t value, sum = 0.0, sq_sum = 0.0;
    mp_obj_iter_buf_t iter_buf;
    mp_obj_t item, iterable = mp_getiter(oin, &iter_buf);
    mp_int_t len = mp_obj_get_int(mp_obj_len(oin));
    while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
        value = mp_obj_get_float(item);
        sum += value;
    }
    if(optype ==  NUMERICAL_SUM) {
        return mp_obj_new_float(sum);
    } else if(optype == NUMERICAL_MEAN) {
        return mp_obj_new_float(sum/len);
    } else { // this should be the case of the standard deviation
            // TODO: implement the Welford algorithm here
        sum /= len; // this is the mean now
        iterable = mp_getiter(oin, &iter_buf);
        while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
            value = mp_obj_get_float(item) - sum;
            sq_sum += value * value;
        }
        return mp_obj_new_float(MICROPY_FLOAT_C_FUN(sqrt)(sq_sum/(len-ddof)));
    }
}

STATIC mp_obj_t numerical_function(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args, uint8_t optype) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_PTR(&mp_const_none_obj)} } ,
        { MP_QSTR_axis, MP_ARG_OBJ, {.u_rom_obj = MP_ROM_PTR(&mp_const_none_obj)} },
    };
    // for this case, the axis keyword will simply be ignored
    
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    
    mp_obj_t oin = args[0].u_obj;
    
    if(MP_OBJ_IS_TYPE(oin, &mp_type_tuple) || MP_OBJ_IS_TYPE(oin, &mp_type_list) || 
        MP_OBJ_IS_TYPE(oin, &mp_type_range)) {
        switch(optype) {
            case NUMERICAL_MIN:
            case NUMERICAL_ARGMIN:
            case NUMERICAL_MAX:
            case NUMERICAL_ARGMAX:
                return numerical_argmin_argmax_iterable(oin, optype);
            case NUMERICAL_SUM:
            case NUMERICAL_MEAN:
                return numerical_sum_mean_std_iterable(oin, optype, 0);
            default: // we should never end up here
                return mp_const_none;
        }
    } else if(MP_OBJ_IS_TYPE(oin, &ulab_ndarray_type)) {
        ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(oin);
        switch(optype) {
            case NUMERICAL_MIN:
            case NUMERICAL_ARGMIN:
            case NUMERICAL_MAX:
            case NUMERICAL_ARGMAX:
                return numerical_argmin_argmax_ndarray(ndarray, optype);
            case NUMERICAL_SUM:
            case NUMERICAL_MEAN:
                return numerical_sum_mean_std_ndarray(ndarray, optype, 0);
            default:
                return mp_const_none;
        }
    } else {
        mp_raise_TypeError(translate("input must be a micropython iterable or an ndarray"));
    }
    return mp_const_none;
}

mp_obj_t numerical_sum(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return numerical_function(n_args, pos_args, kw_args, NUMERICAL_SUM);
}

mp_obj_t numerical_mean(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return numerical_function(n_args, pos_args, kw_args, NUMERICAL_MEAN);
}

mp_obj_t numerical_std(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_PTR(&mp_const_none_obj)} } ,
        { MP_QSTR_axis, MP_ARG_OBJ, {.u_rom_obj = MP_ROM_PTR(&mp_const_none_obj)} },
        { MP_QSTR_ddof, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = 0} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    
    mp_obj_t oin = args[0].u_obj;
    if(MP_OBJ_IS_TYPE(oin, &mp_type_tuple) 
       || MP_OBJ_IS_TYPE(oin, &mp_type_list) 
       || MP_OBJ_IS_TYPE(oin, &mp_type_range)) {
        return numerical_sum_mean_std_iterable(oin, NUMERICAL_STD, args[2].u_int);
    } else if(MP_OBJ_IS_TYPE(oin, &ulab_ndarray_type)) {
        ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(oin);
        if(args[2].u_int > ndarray->len-1) {
            mp_raise_ValueError(translate("degrees of freedom is larger then length of array"));
        }
        return numerical_sum_mean_std_ndarray(ndarray, NUMERICAL_STD, args[2].u_int);
    } else {
        mp_raise_TypeError(translate("input must be tuple, list, range, or ndarray"));
    }
    return mp_const_none;
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_linspace_obj, 2, numerical_linspace);
MP_DEFINE_CONST_FUN_OBJ_KW(numerical_sum_obj, 1, numerical_sum);
MP_DEFINE_CONST_FUN_OBJ_KW(numerical_mean_obj, 1, numerical_mean);
MP_DEFINE_CONST_FUN_OBJ_KW(numerical_std_obj, 1, numerical_std);
