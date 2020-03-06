
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

#if ULAB_NUMERICAL_MODULE

enum NUMERICAL_FUNCTION_TYPE {
    NUMERICAL_MIN,
    NUMERICAL_MAX,
    NUMERICAL_ARGMIN,
    NUMERICAL_ARGMAX,
    NUMERICAL_SUM,
    NUMERICAL_MEAN,
    NUMERICAL_STD,
};

ndarray_header_obj_t contracted_shape_strides(ndarray_obj_t *ndarray, int8_t axis) {
	// creates the shape and strides arrays of the contracted ndarray, 
	// i.e., when one of the axes is removed
	// wrap around, if the axis is negative
    if(axis < 0) axis += ndarray->ndim;
    if((axis > ndarray->ndim-1) || (axis < 0)) {
        mp_raise_ValueError("tuple index out of range");
    }
    size_t *shape = m_new(size_t, ndarray->ndim-1);
    int32_t *strides = m_new(int32_t, ndarray->ndim-1);
    for(size_t i=0, j=0; i < ndarray->ndim; i++) {
        if(axis != i) {
            shape[j] = ndarray->shape[j];
            j++;
        }
    }
    int32_t stride = 1;
    for(size_t i=0; i < ndarray->ndim-1; i++) {
        strides[ndarray->ndim-2-i] = stride;
        stride *= shape[ndarray->ndim-2-i];
    }
    ndarray_header_obj_t header;
    header.shape = shape;
    header.strides = strides;
    header.axis = axis;
    return header;
}

// numerical functions for ndarrays
mp_obj_t numerical_flat_sum_mean_std(ndarray_obj_t *ndarray, uint8_t optype, size_t ddof) {
	if(ndarray->len <= ddof) {
		mp_raise_ValueError(translate("ndarray too short for ddof"));		
	}
	mp_float_t value;
	size_t *coords = ndarray_new_coords(ndarray->ndim);
	if(ndarray->dtype == NDARRAY_UINT8) {
		CALCULATE_FLAT_SUM_STD(ndarray, coords, uint8_t, value, optype);
	} else if(ndarray->dtype == NDARRAY_INT8) {
		CALCULATE_FLAT_SUM_STD(ndarray, coords, int8_t, value, optype);
	} if(ndarray->dtype == NDARRAY_UINT16) {
		CALCULATE_FLAT_SUM_STD(ndarray, coords, uint16_t, value, optype);
	} else if(ndarray->dtype == NDARRAY_INT16) {
		CALCULATE_FLAT_SUM_STD(ndarray, coords, int16_t, value, optype);
	} else {
		CALCULATE_FLAT_SUM_STD(ndarray, coords, mp_float_t, value, optype);
	}
	if(optype == NUMERICAL_SUM) {
		return mp_obj_new_float(value);
	} else if(optype == NUMERICAL_MEAN) {
		return mp_obj_new_float(value/ndarray->len);
	} else {
		return mp_obj_new_float(MICROPY_FLOAT_C_FUN(sqrt)(value/(ndarray->len-ddof)));
	}
}   

mp_obj_t numerical_sum_mean_std_ndarray(ndarray_obj_t *ndarray, mp_obj_t axis, uint8_t optype, size_t ddof) {
	if(axis == mp_const_none) {
		return numerical_flat_sum_mean_std(ndarray, optype, ddof);
	} /*else {
		int8_t ax = mp_obj_get_int(axis);
		ndarray_header_obj_t header = contracted_shape_strides(ndarray, ax);
		ndarray_obj_t *result = ndarray_new_ndarray(ndarray->ndim-1, header.shape, header.strides, NDARRAY_FLOAT);
		mp_float_t *farray = (mp_float_t *)result->array->items;
		size_t offset;
		// iterate along the length of the output array, so as to avoid recursion
		for(size_t i=0; i < result->array->len; i++) {
			offset = ndarray_index_from_contracted(i, ndarray, result->strides, result->ndim, header.axis) + ndarray->offset;
			if(ndarray->array->typecode == NDARRAY_UINT8) {
				CALCULATE_SUM(ndarray, uint8_t, farray, ndarray->shape[header.axis], i, ndarray->strides[header.axis], offset, optype);
			} else if(ndarray->array->typecode == NDARRAY_INT8) {
				CALCULATE_SUM(ndarray, int8_t, farray, ndarray->shape[header.axis], i, ndarray->strides[header.axis], offset, optype); 
			} else if(ndarray->array->typecode == NDARRAY_UINT16) {
				CALCULATE_SUM(ndarray, uint16_t, farray, ndarray->shape[header.axis], i, ndarray->strides[header.axis], offset, optype);     
			} else if(ndarray->array->typecode == NDARRAY_INT16) {
				CALCULATE_SUM(ndarray, int16_t, farray, ndarray->shape[header.axis], i, ndarray->strides[header.axis], offset, optype);
			} else {
				CALCULATE_SUM(ndarray, mp_float_t, farray, ndarray->shape[header.axis], i, ndarray->strides[header.axis], offset, optype);       
			}
			if(optype == NUMERICAL_MEAN) farray[i] /= ndarray->shape[header.axis];
		}
		return MP_OBJ_FROM_PTR(result);
	}*/
	return mp_const_none;
}

mp_obj_t numerical_argmin_argmax_ndarray(ndarray_obj_t *ndarray, mp_obj_t axis, uint8_t optype) {
    return mp_const_none;
}


// numerical functions for general python iterables
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
	// this function is called, when the argument of sum/mean/std is a standard python iterable
    mp_float_t value, sum = 0.0, mtmp = 0.0, m = 0.0;
    mp_obj_iter_buf_t iter_buf;
    mp_obj_t item, iterable = mp_getiter(oin, &iter_buf);
	if(mp_obj_get_int(mp_obj_len(oin)) <= ddof) {
		mp_raise_ValueError(translate("iterable too short for ddof"));
	}
    size_t len = 0;
    while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
        value = mp_obj_get_float(item);
        if(optype == NUMERICAL_STD) {
			mtmp = value;
			m = mtmp + (value - mtmp) / (len+1);
			sum += (value - mtmp) * (value - m);
			len++;
		} else {
			sum += value;
		}
    }
    if(optype ==  NUMERICAL_SUM) {
        return mp_obj_new_float(sum);
    } else if(optype == NUMERICAL_MEAN) {
        return mp_obj_new_float(sum/len);
    } else { // i.e., optype = NUMERICAL_STD
        return mp_obj_new_float(MICROPY_FLOAT_C_FUN(sqrt)(sum/(len-ddof)));
    }
}

STATIC mp_obj_t numerical_sum_mean_std_ndarray(ndarray_obj_t *ndarray, mp_obj_t axis, uint8_t optype, size_t ddof) {
    size_t m, n, increment, start, start_inc, N, len; 
    axis_sorter(ndarray, axis, &m, &n, &N, &increment, &len, &start_inc);
    ndarray_obj_t *results = create_new_ndarray(m, n, NDARRAY_FLOAT);
    mp_float_t sum, sq_sum;
    mp_float_t *farray = (mp_float_t *)results->array->items;
    for(size_t j=0; j < N; j++) { // result index
        start = j * start_inc;
        sum = sq_sum = 0.0;
        if(ndarray->dtype == NDARRAY_UINT8) {
            RUN_SUM(ndarray, uint8_t, optype, len, start, increment);
        } else if(ndarray->dtype == NDARRAY_INT8) {
            RUN_SUM(ndarray, int8_t, optype, len, start, increment);
        } else if(ndarray->dtype == NDARRAY_UINT16) {
            RUN_SUM(ndarray, uint16_t, optype, len, start, increment);
        } else if(ndarray->dtype == NDARRAY_INT16) {
            RUN_SUM(ndarray, int16_t, optype, len, start, increment);
        } else { // this will be mp_float_t, no need to check
            RUN_SUM(ndarray, mp_float_t, optype, len, start, increment);
        }
        if(optype == NUMERICAL_SUM) {
            farray[j] = sum;
        } else { // this is the case of the mean
            farray[j] = sum / len;
        }
    }
    if(results->array->len == 1) {
        return mp_obj_new_float(farray[0]);
    }
    return MP_OBJ_FROM_PTR(results);
}

/*
mp_obj_t numerical_std_ndarray(ndarray_obj_t *ndarray, mp_obj_t axis, size_t ddof) {
    size_t m, n, increment, start, start_inc, N, len; 
    mp_float_t sum, sum_sq;
    
    axis_sorter(ndarray, axis, &m, &n, &N, &increment, &len, &start_inc);
    if(ddof > len) {
        mp_raise_ValueError(translate("ddof must be smaller than length of data set"));
    }
    ndarray_obj_t *results = create_new_ndarray(m, n, NDARRAY_FLOAT);
    mp_float_t *farray = (mp_float_t *)results->array->items;
    for(size_t j=0; j < N; j++) { // result index
        start = j * start_inc;
        sum = 0.0;
        sum_sq = 0.0;
        if(ndarray->array->dtype == NDARRAY_UINT8) {
            RUN_STD(ndarray, uint8_t, len, start, increment);
        } else if(ndarray->array->dtype == NDARRAY_INT8) {
            RUN_STD(ndarray, int8_t, len, start, increment);
        } else if(ndarray->array->dtype == NDARRAY_UINT16) {
            RUN_STD(ndarray, uint16_t, len, start, increment);
        } else if(ndarray->array->dtype == NDARRAY_INT16) {
            RUN_STD(ndarray, int16_t, len, start, increment);
        } else { // this will be mp_float_t, no need to check
            RUN_STD(ndarray, mp_float_t, len, start, increment);
        }
        farray[j] = MICROPY_FLOAT_C_FUN(sqrt)(sum_sq/(len - ddof));
    }
    if(results->array->len == 1) {
        return mp_obj_new_float(farray[0]);
    }
    return MP_OBJ_FROM_PTR(results);
}
*/

mp_obj_t numerical_argmin_argmax_iterable(mp_obj_t oin, mp_obj_t axis, uint8_t optype) {
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

mp_obj_t numerical_argmin_argmax_ndarray(ndarray_obj_t *ndarray, mp_obj_t axis, uint8_t optype) {
    size_t m, n, increment, start, start_inc, N, len;
    axis_sorter(ndarray, axis, &m, &n, &N, &increment, &len, &start_inc);
    ndarray_obj_t *results;
    if((optype == NUMERICAL_ARGMIN) || (optype == NUMERICAL_ARGMAX)) {
        // we could save some RAM by taking NDARRAY_UINT8, if the dimensions 
        // are smaller than 256, but the code would become more involving 
        // (we would also need extra flash space)
        results = create_new_ndarray(m, n, NDARRAY_UINT16);
    } else {
        results = create_new_ndarray(m, n, ndarray->array->dtype);
    }
    
    for(size_t j=0; j < N; j++) { // result index
        start = j * start_inc;
        if((ndarray->array->dtype == NDARRAY_UINT8) || (ndarray->array->dtype == NDARRAY_INT8)) {
            if((optype == NUMERICAL_MAX) || (optype == NUMERICAL_MIN)) {
                RUN_ARGMIN(ndarray, results, uint8_t, uint8_t, len, start, increment, optype, j);
            } else {
                RUN_ARGMIN(ndarray, results, uint8_t, uint16_t, len, start, increment, optype, j);                
            }
        } else if((ndarray->array->dtype == NDARRAY_UINT16) || (ndarray->array->dtype == NDARRAY_INT16)) {
            RUN_ARGMIN(ndarray, results, uint16_t, uint16_t, len, start, increment, optype, j);
        } else {
            if((optype == NUMERICAL_MAX) || (optype == NUMERICAL_MIN)) {
                RUN_ARGMIN(ndarray, results, mp_float_t, mp_float_t, len, start, increment, optype, j);
            } else {
                RUN_ARGMIN(ndarray, results, mp_float_t, uint16_t, len, start, increment, optype, j);                
            }
        }
    }
    return MP_OBJ_FROM_PTR(results);
}

STATIC mp_obj_t numerical_function(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args, uint8_t optype) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = mp_const_none} } ,
        { MP_QSTR_axis, MP_ARG_OBJ, {.u_rom_obj = mp_const_none } },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    
    mp_obj_t oin = args[0].u_obj;
    mp_obj_t axis = args[1].u_obj;
    if((axis != mp_const_none) && (mp_obj_get_int(axis) != 0) && (mp_obj_get_int(axis) != 1)) {
        // this seems to pass with False, and True...
        mp_raise_ValueError(translate("axis must be None, 0, or 1"));
    }
    
    if(MP_OBJ_IS_TYPE(oin, &mp_type_tuple) || MP_OBJ_IS_TYPE(oin, &mp_type_list) || 
        MP_OBJ_IS_TYPE(oin, &mp_type_range)) {
        switch(optype) {
            case NUMERICAL_MIN:
            case NUMERICAL_ARGMIN:
            case NUMERICAL_MAX:
            case NUMERICAL_ARGMAX:
                return numerical_argmin_argmax_iterable(oin, axis, optype);
            case NUMERICAL_SUM:
            case NUMERICAL_MEAN:
                return numerical_sum_mean_std_iterable(oin, optype, 0);
            default: // we should never reach this point, but whatever
                return mp_const_none;
        }
    } else if(MP_OBJ_IS_TYPE(oin, &ulab_ndarray_type)) {
        ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(oin);
        switch(optype) {
            case NUMERICAL_MIN:
            case NUMERICAL_MAX:
            case NUMERICAL_ARGMIN:
            case NUMERICAL_ARGMAX:
                return numerical_argmin_argmax_ndarray(ndarray, axis, optype);
            case NUMERICAL_SUM:
            case NUMERICAL_MEAN:
                return numerical_sum_mean_ndarray(ndarray, axis, optype);
            default:
                mp_raise_NotImplementedError(translate("operation is not implemented on ndarrays"));
        }
    } else {
        mp_raise_TypeError(translate("input must be tuple, list, range, or ndarray"));
    }
    return mp_const_none;
}

mp_obj_t numerical_min(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return numerical_function(n_args, pos_args, kw_args, NUMERICAL_MIN);
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_min_obj, 1, numerical_min);

mp_obj_t numerical_max(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return numerical_function(n_args, pos_args, kw_args, NUMERICAL_MAX);
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_max_obj, 1, numerical_max);

mp_obj_t numerical_argmin(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return numerical_function(n_args, pos_args, kw_args, NUMERICAL_ARGMIN);
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_argmin_obj, 1, numerical_argmin);

mp_obj_t numerical_argmax(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return numerical_function(n_args, pos_args, kw_args, NUMERICAL_ARGMAX);
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_argmax_obj, 1, numerical_argmax);

mp_obj_t numerical_sum(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return numerical_function(n_args, pos_args, kw_args, NUMERICAL_SUM);
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_sum_obj, 1, numerical_sum);

mp_obj_t numerical_mean(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return numerical_function(n_args, pos_args, kw_args, NUMERICAL_MEAN);
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_mean_obj, 1, numerical_mean);

mp_obj_t numerical_std(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = mp_const_none } } ,
        { MP_QSTR_axis, MP_ARG_OBJ, {.u_rom_obj = mp_const_none } },
        { MP_QSTR_ddof, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = 0} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    
    mp_obj_t oin = args[0].u_obj;
    mp_obj_t axis = args[1].u_obj;
    size_t ddof = args[2].u_int;
    if((axis != mp_const_none) && (mp_obj_get_int(axis) != 0) && (mp_obj_get_int(axis) != 1)) {
        // this seems to pass with False, and True...
        mp_raise_ValueError(translate("axis must be None, 0, or 1"));
    }
    if(MP_OBJ_IS_TYPE(oin, &mp_type_tuple) || MP_OBJ_IS_TYPE(oin, &mp_type_list) || MP_OBJ_IS_TYPE(oin, &mp_type_range)) {
        return numerical_sum_mean_std_iterable(oin, NUMERICAL_STD, ddof);
    } else if(MP_OBJ_IS_TYPE(oin, &ulab_ndarray_type)) {
        ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(oin);
        return numerical_std_ndarray(ndarray, axis, ddof);
    } else {
        mp_raise_TypeError(translate("input must be tuple, list, range, or ndarray"));
    }
    return mp_const_none;
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_std_obj, 1, numerical_std);

mp_obj_t numerical_diff(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = mp_const_none } },
        { MP_QSTR_n, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = 1 } },
        { MP_QSTR_axis, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = -1 } },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(1, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    
	if(args[1].u_int != 1) {
        mp_raise_NotImplementedError(translate("only first derivatives implemented"));
	}
	if(MP_OBJ_IS_TYPE(oin, &mp_type_tuple) || MP_OBJ_IS_TYPE(oin, &mp_type_list) || 
        MP_OBJ_IS_TYPE(oin, &mp_type_range)) {
		size_t len = mp_obj_get_int(mp_obj_len(args[0].u_obj));
		if(len < 2) {
			mp_raise_ValueError(translate("argument too short"));
		}
		ndarray_obj_t *ndarray = ndarray_new_linear_array(len, NDARRAY_FLOAT);
		mp_float_t *array = (mp_float_t *)ndarray->array;
		mp_obj_iter_buf_t iter_buf;
		mp_obj_t iterable = mp_getiter(oin, &iter_buf);
		mp_obj_t item;
		item = mp_iternext(iterable);
		mp_float_t value = mp_obj_get_float(item);
		mp_float_t nvalue;
		while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
			nvalue = mp_obj_get_float(item);
			*array++ = nvalue - value;
			value = nvalue;
		}
		return MP_OBJ_FROM_PTR(ndarray);
	}
    if(!MP_OBJ_IS_TYPE(args[0].u_obj, &ulab_ndarray_type)) {
        mp_raise_TypeError(translate("wrong input type"));
    }
    
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(args[0].u_obj);
    uint8_t axis = args[2].u_int;
    if(axis < 0) axis += ndarray->ndim;
    if((axis > ndarray->ndim-1) || (axis < 0)) {
        mp_raise_ValueError(translate("tuple index out of range"));
    }
    if(ndarray->shape[axis] == 1) {
		mp_raise_ValueError(translate("axis is too short"));
	}
    // the shape along 'axis' is reduced by 1
	size_t *shape = m_new(size_t, ndarray->ndim);
    for(uint8_t i=0; i < ndarray->ndim; i++) {
		shape[i] = ndarray->shape[i];
		if(i == axis) shape[i]--;
	}
    ndarray_obj_t *result = ndarray_new_dense_ndarray(ndarray->ndim, shape, ndarray->dtype);
    m_del(size_t, shape, ndarray->ndim);
	size_t *coords = ndarray_new_coords(ndarray->ndim-1);
	int32_t *strides = ndarray_contract_strides(ndarray, axis);
	size_t *shape = ndarray_contract_shape(ndarray, axis);
	
    if(ndarray->dtype == NDARRAY_UINT8) {
        CALCULATE_DIFF(ndarray, result, uint8_t, axis, coords, shapes, strides);
    } else if(ndarray->dtype == NDARRAY_INT8) {
        CALCULATE_DIFF(ndarray, result, int8_t, axis, coords, shapes, strides);
    }  else if(ndarray->dtype == NDARRAY_UINT16) {
        CALCULATE_DIFF(ndarray, result, uint16_t, axis, coords, shapes, strides);
    } else if(ndarray->dtype == NDARRAY_INT16) {
        CALCULATE_DIFF(ndarray, result, int16_t, axis, coords, shapes, strides);
    } else { // if we got this far, the dtype must be float
        CALCULATE_DIFF(ndarray, result, mp_float_t, axis, coords, shapes, strides);
    }
	m_del(size_t, coords, ndarray->ndim-1);
	m_del(size_t, strides, ndarray->ndim-1);
	m_del(size_t, shape, ndarray->ndim-1);		
    return MP_OBJ_FROM_PTR(result);
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_diff_obj, 1, numerical_diff);
/*
mp_obj_t numerical_sort_helper(mp_obj_t oin, mp_obj_t axis, uint8_t inplace) {
    if(!MP_OBJ_IS_TYPE(oin, &ulab_ndarray_type)) {
        mp_raise_TypeError(translate("sort argument must be an ndarray"));
    }

    ndarray_obj_t *ndarray;
    mp_obj_t out;
    if(inplace == 1) {
        ndarray = MP_OBJ_TO_PTR(oin);
    } else {
        out = ndarray_copy(oin);
        ndarray = MP_OBJ_TO_PTR(out);
    }
    size_t increment, start_inc, end, N;
    if(axis == mp_const_none) { // flatten the array
        ndarray->m = 1;
        ndarray->n = ndarray->array->len;
        increment = 1;
        start_inc = ndarray->n;
        end = ndarray->n;
        N = ndarray->n;
    } else if((mp_obj_get_int(axis) == -1) || 
              (mp_obj_get_int(axis) == 1)) { // sort along the horizontal axis
        increment = 1;
        start_inc = ndarray->n;
        end = ndarray->array->len;
        N = ndarray->n;
    } else if(mp_obj_get_int(axis) == 0) { // sort along vertical axis
        increment = ndarray->n;
        start_inc = 1;
        end = ndarray->m;
        N = ndarray->m;
    } else {
        mp_raise_ValueError(translate("axis must be -1, 0, None, or 1"));
    }
    
    size_t q, k, p, c;

    for(size_t start=0; start < end; start+=start_inc) {
        q = N; 
        k = (q >> 1);
        if((ndarray->dtype == NDARRAY_UINT8) || (ndarray->dtype == NDARRAY_INT8)) {
            HEAPSORT(uint8_t, ndarray);
        } else if((ndarray->dtype == NDARRAY_INT16) || (ndarray->dtype == NDARRAY_INT16)) {
            HEAPSORT(uint16_t, ndarray);
        } else {
            HEAPSORT(mp_float_t, ndarray);
        }
    }
    if(inplace == 1) {
        return mp_const_none;
    } else {
        return out;
    }
}

// numpy function
mp_obj_t numerical_sort(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = mp_const_none } },
        { MP_QSTR_axis, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_int = -1 } },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(1, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    return numerical_sort_helper(args[0].u_obj, args[1].u_obj, 0);
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_sort_obj, 1, numerical_sort);

// method of an ndarray
mp_obj_t numerical_sort_inplace(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = mp_const_none } },
        { MP_QSTR_axis, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_int = -1 } },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(1, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    return numerical_sort_helper(args[0].u_obj, args[1].u_obj, 1);
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_sort_inplace_obj, 1, numerical_sort_inplace);


mp_obj_t numerical_argsort(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = mp_const_none } },
        { MP_QSTR_axis, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_int = -1 } },
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(1, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    if(!MP_OBJ_IS_TYPE(args[0].u_obj, &ulab_ndarray_type)) {
        mp_raise_TypeError(translate("argsort argument must be an ndarray"));
    }

    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(args[0].u_obj);
    size_t increment, start_inc, end, N, m, n;
    if(args[1].u_obj == mp_const_none) { // flatten the array
        m = 1;
        n = ndarray->array->len;
        ndarray->m = m;
        ndarray->n = n;
        increment = 1;
        start_inc = ndarray->n;
        end = ndarray->n;
        N = n;
    } else if((mp_obj_get_int(args[1].u_obj) == -1) || 
              (mp_obj_get_int(args[1].u_obj) == 1)) { // sort along the horizontal axis
        m = ndarray->m;
        n = ndarray->n;
        increment = 1;
        start_inc = n;
        end = ndarray->array->len;
        N = n;
    } else if(mp_obj_get_int(args[1].u_obj) == 0) { // sort along vertical axis
        m = ndarray->m;
        n = ndarray->n;
        increment = n;
        start_inc = 1;
        end = m;
        N = m;
    } else {
        mp_raise_ValueError(translate("axis must be -1, 0, None, or 1"));
    }

    // at the expense of flash, we could save RAM by creating 
    // an NDARRAY_UINT16 ndarray only, if needed, otherwise, NDARRAY_UINT8
    ndarray_obj_t *indices = create_new_ndarray(m, n, NDARRAY_UINT16);
    uint16_t *index_array = (uint16_t *)indices->array;
    // initialise the index array
    // if array is flat: 0 to indices->n
    // if sorting vertically, identical indices are arranged row-wise
    // if sorting horizontally, identical indices are arranged colunn-wise
    for(uint16_t start=0; start < end; start+=start_inc) {
        for(uint16_t s=0; s < N; s++) {
            index_array[start+s*increment] = s;
        }
    }

    size_t q, k, p, c;
    for(size_t start=0; start < end; start+=start_inc) {
        q = N; 
        k = (q >> 1);
        if((ndarray->dtype == NDARRAY_UINT8) || (ndarray->dtype == NDARRAY_INT8)) {
            HEAP_ARGSORT(uint8_t, ndarray, index_array);
        } else if((ndarray->dtype == NDARRAY_INT16) || (ndarray->dtype == NDARRAY_INT16)) {
            HEAP_ARGSORT(uint16_t, ndarray, index_array);
        } else {
            HEAP_ARGSORT(mp_float_t, ndarray, index_array);
        }
    }
    return MP_OBJ_FROM_PTR(indices);
}

MP_DEFINE_CONST_FUN_OBJ_KW(numerical_argsort_obj, 1, numerical_argsort);
*/
STATIC const mp_rom_map_elem_t ulab_numerical_globals_table[] = {
    { MP_OBJ_NEW_QSTR(MP_QSTR_sum), (mp_obj_t)&numerical_sum_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_mean), (mp_obj_t)&numerical_mean_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_std), (mp_obj_t)&numerical_std_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_min), (mp_obj_t)&numerical_min_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_max), (mp_obj_t)&numerical_max_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_argmin), (mp_obj_t)&numerical_argmin_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_argmax), (mp_obj_t)&numerical_argmax_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_diff), (mp_obj_t)&numerical_diff_obj },
//    { MP_OBJ_NEW_QSTR(MP_QSTR_sort), (mp_obj_t)&numerical_sort_obj },
//    { MP_OBJ_NEW_QSTR(MP_QSTR_argsort), (mp_obj_t)&numerical_argsort_obj },    
};

STATIC MP_DEFINE_CONST_DICT(mp_module_ulab_numerical_globals, ulab_numerical_globals_table);

mp_obj_module_t ulab_numerical_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&mp_module_ulab_numerical_globals,
};

#endif
