
/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 Jeff Epler for Adafruit Industries
 * Copyright (c) 2019-2020 Zoltán Vörös
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "py/runtime.h"
#include "py/binary.h"
#include "py/obj.h"
#include "py/objtuple.h"
#include "ndarray.h"
#include "ndarray_properties.h"


mp_float_t ndarray_get_float_value(void *data, uint8_t dtype, size_t index) {
    if(dtype == NDARRAY_UINT8) {
        return (mp_float_t)((uint8_t *)data)[index];
    } else if(dtype == NDARRAY_INT8) {
        return (mp_float_t)((int8_t *)data)[index];
    } else if(dtype == NDARRAY_UINT16) {
        return (mp_float_t)((uint16_t *)data)[index];
    } else if(dtype == NDARRAY_INT16) {
        return (mp_float_t)((int16_t *)data)[index];
    } else {
        return (mp_float_t)((mp_float_t *)data)[index];
    }
}

void ndarray_fill_array_iterable(mp_float_t *array, mp_obj_t iterable) {
    mp_obj_iter_buf_t x_buf;
    mp_obj_t x_item, x_iterable = mp_getiter(iterable, &x_buf);
    size_t i=0;
    while ((x_item = mp_iternext(x_iterable)) != MP_OBJ_STOP_ITERATION) {
        *array++ = (mp_float_t)mp_obj_get_float(x_item);
        i++;
    }
}

int32_t *strides_from_shape(size_t *shape, size_t n) {
    // returns a strides array that corresponds to a dense array with the prescribed shape
    int32_t *strides = m_new(int32_t, n);
    strides[n-1] = 1;
    for(uint8_t i=n-1; i > 0; i--) {
        strides[i-1] = strides[i] * shape[i];
    }
    return strides;
}

size_t *ndarray_new_coords(uint8_t ndim) {
    size_t *coords = m_new(size_t, ndim);
    memset(coords, 0, ndim*sizeof(size_t));
    return coords;
}

size_t *ndarray_contract_shape(ndarray_obj_t *ndarray, uint8_t axis) {
	// removes a single axis from the shape array
	if(ndarray->shape[axis] == 1) {
		mp_raise_ValueError(translate("tensor cannot be contracted along axis"));
	}
	size_t *shape = m_new(size_t, ndarray->ndim-1);	
	uint8_t j = 0;
	for(uint8_t i=0; i < ndarray->ndim; i++) {
		if(axis != i) {
			shape[j] = ndarray->shape[i];
			j++;
		}
	}
	return shape;
}
int32_t *ndarray_contract_strides(ndarray_obj_t *ndarray, uint8_t axis) {
	// removes a single axis from the strides array
	if(ndarray->shape[axis] == 1) {
		mp_raise_ValueError(translate("tensor cannot be contracted along axis"));
	}
	int32_t *strides = m_new(int32_t, ndarray->ndim-1);
	uint8_t j = 0;
	for(uint8_t i=0; i < ndarray->ndim; i++) {
		if(axis != i) {
			strides[j] = ndarray->strides[i];
			j++;
		}
	}
	return strides;
}

void ndarray_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind) {
    (void)kind;
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    uint8_t print_extra = self->ndim;
    size_t *coords = ndarray_new_coords(self->ndim);
    int32_t last_stride = self->strides[self->ndim-1];
    
    size_t offset = 0;
    if(self->len == 0) mp_print_str(print, "array([");
    for(size_t i=0; i < self->len; i++) {
        for(uint8_t j=0; j < print_extra; j++) {
            mp_print_str(print, "[");
        }
        print_extra = 0;
        if(!self->boolean) {
            mp_obj_print_helper(print, mp_binary_get_val_array(self->dtype, self->array, offset), PRINT_REPR);
        } else {
            if(((uint8_t *)self->array)[offset]) {
                mp_print_str(print, "True");
            } else {
                mp_print_str(print, "False");
            }
        }
		offset += last_stride;
        coords[self->ndim-1] += 1;
        if(coords[self->ndim-1] != self->shape[self->ndim-1]) {
            mp_print_str(print, ", ");
        }
        for(uint8_t j=self->ndim-1; j > 0; j--) {
            if(coords[j] == self->shape[j]) {
                offset -= self->shape[j] * self->strides[j];
                offset += self->strides[j-1];
                print_extra += 1;
                coords[j] = 0;
                coords[j-1] += 1;
                mp_print_str(print, "]");
            } else { // coordinates can change only, if the last coordinate changes
                break;
            }
        }
        if(print_extra && (i != self->len-1)) {
            mp_print_str(print, "\n");
            if(print_extra > 1) {
                mp_print_str(print, "\n");
            }
        }
    }
    m_del(size_t, coords, self->ndim);
	mp_print_str(print, "]");
	if(self->boolean) {
		mp_print_str(print, ", dtype=bool)");
	} else if(self->dtype == NDARRAY_UINT8) {
		mp_print_str(print, ", dtype=uint8)");
	} else if(self->dtype == NDARRAY_INT8) {
		mp_print_str(print, ", dtype=int8)");
	} else if(self->dtype == NDARRAY_UINT16) {
		mp_print_str(print, ", dtype=uint16)");
	} else if(self->dtype == NDARRAY_INT16) {
		mp_print_str(print, ", dtype=int16)");
	} else if(self->dtype == NDARRAY_FLOAT) {
		mp_print_str(print, ", dtype=float)");
	}
}

void ndarray_assign_elements(ndarray_obj_t *ndarray, mp_obj_t iterable, uint8_t dtype, size_t *idx) {
    // assigns a single row in the matrix
    mp_obj_t item;
    uint8_t *array = (uint8_t *)ndarray->array;
    array += *idx;
	if(ndarray->boolean) {
	    while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
            // TODO: this might be wrong here: we have to check for the trueness of item
            if(mp_obj_is_true(item)) {
                *array = 1;
            }
            array++;
            (*idx)++;
        }
    } else {
        while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
            mp_binary_set_val_array(dtype, ndarray->array, (*idx)++, item);
        }
    }
}

bool ndarray_is_dense(ndarray_obj_t *ndarray) {
	// returns true, if the array is dense, false otherwise
	// the array should dense, if the very first stride can be calculated from shape
	size_t stride = 1;
	for(uint8_t i=0; i < ndarray->ndim; i++) {
        stride *= ndarray->shape[i];
    }
	return stride == ndarray->strides[0] ? true : false;
}
 
ndarray_obj_t *ndarray_new_ndarray(uint8_t ndim, size_t *shape, int32_t *strides, uint8_t dtype) {
    // Creates the base ndarray with shape, and initialises the values to straight 0s
    // the function should work in the general n-dimensional case
    ndarray_obj_t *ndarray = m_new_obj(ndarray_obj_t);
    ndarray->base.type = &ulab_ndarray_type;
    ndarray->dtype = dtype;
    ndarray->ndim = ndim;
	ndarray->len = 1;
    for(uint8_t i=0; i < ndim; i++) {
		ndarray->shape[i] = shape[i];
		ndarray->strides[i] = strides[i];
		ndarray->len *= shape[i];
	}
    if(dtype == NDARRAY_BOOL) {
        dtype = NDARRAY_UINT8;
        ndarray->boolean = NDARRAY_BOOLEAN;
    } else {
        ndarray->boolean = NDARRAY_NUMERIC;
    }
	uint8_t itemsize = mp_binary_get_size('@', dtype, NULL);
	uint8_t *array = m_new(byte, itemsize*ndarray->len);
    // this should set all elements to 0, irrespective of the of the dtype (all bits are zero)
    // we could, perhaps, leave this step out, and initialise the array only, when needed
    memset(array, 0, ndarray->len*itemsize);
    ndarray->array = array;
    return ndarray;
}

ndarray_obj_t *ndarray_new_dense_ndarray(uint8_t ndim, size_t *shape, uint8_t dtype) {
    // creates a dense array, i.e., one, where the strides are derived directly from the shapes
    // the function should work in the general n-dimensional case
    int32_t *strides = m_new(int32_t, ndim);
    strides[ndim-1] = 1;
    for(size_t i=ndim-1; i > 0; i--) {
        strides[i-1] = strides[i] * shape[i-1];
    }
    return ndarray_new_ndarray(ndim, shape, strides, dtype);
}

ndarray_obj_t *ndarray_new_ndarray_from_tuple(mp_obj_tuple_t *_shape, uint8_t dtype) {
    // creates a dense array from a tuple
    // the function should work in the general n-dimensional case
    uint8_t ndim = _shape->len;
    size_t *shape = m_new(size_t, ndim);
    for(size_t i=0; i < ndim; i++) {
		shape[i] = mp_obj_get_int(_shape->items[i]);
    }
    return ndarray_new_dense_ndarray(ndim, shape, dtype);
}

ndarray_obj_t *ndarray_new_view(ndarray_obj_t *source, uint8_t ndim, size_t *shape, int32_t *strides, int32_t offset) {
    // creates a new view from the input arguments
    // the function should work in the n-dimensional case
    ndarray_obj_t *ndarray = m_new_obj(ndarray_obj_t);
    ndarray->base.type = &ulab_ndarray_type;
    ndarray->boolean = source->boolean;
    ndarray->dtype = source->dtype;
    ndarray->ndim = ndim;
    ndarray->len = 1;
    for(uint8_t i=0; i < ndim; i++) {
		ndarray->shape[i] = shape[i];
		ndarray->strides[i] = strides[i];
        ndarray->len *= shape[i];
    }
	uint8_t itemsize = mp_binary_get_size('@', source->dtype, NULL);
    ndarray->array = (uint8_t *)source->array + offset * itemsize;
    return ndarray;
}

void ndarray_copy_array(ndarray_obj_t *source, ndarray_obj_t *target) {
	// copies the content of source->array into a new dense void pointer
	// it is assumed that the dtypes in source and target are the same
	size_t *coords = ndarray_new_coords(source->ndim);
	int32_t last_stride = source->strides[source->ndim-1];
	uint8_t itemsize = mp_binary_get_size('@', source->dtype, NULL);

	uint8_t *array = (uint8_t *)source->array;
	uint8_t *new_array = (uint8_t *)target->array;

	for(size_t i=0; i < source->len; i++) {
		memcpy(new_array, array, itemsize);
		new_array += itemsize;        
		array += last_stride*itemsize;
		coords[source->ndim-1] += 1;
		for(uint8_t j=source->ndim-1; j > 0; j--) {
			if(coords[j] == source->shape[j]) {
				array -= source->shape[j] * source->strides[j] * itemsize;
				array += source->strides[j-1] * itemsize;
				coords[j] = 0;
				coords[j-1] += 1;
			} else { // coordinates can change only, if the last coordinate changes
				break;
			}
		}
	}
	m_del(size_t, coords, source->ndim);
}

ndarray_obj_t *ndarray_copy_view(ndarray_obj_t *source) {
    // creates a one-to-one deep copy of the input ndarray or its view
    // the function should work in the general n-dimensional case
    // In order to make it dtype-agnostic, we copy the memory content 
    // instead of reading out the values
    
    int32_t *strides = strides_from_shape(source->shape, source->ndim);

    uint8_t dtype = source->dtype;
    if(source->boolean) {
        dtype = NDARRAY_BOOLEAN;
    }
    ndarray_obj_t *ndarray = ndarray_new_ndarray(source->ndim, source->shape, strides, dtype);
	ndarray_copy_array(source, ndarray);
    return ndarray;
}

ndarray_obj_t *ndarray_new_linear_array(size_t len, uint8_t dtype) {
    size_t *shape = m_new(size_t, 1);
    int32_t *strides = m_new(int32_t, 1);
    shape[0] = len;
    strides[0] = 1;
    return ndarray_new_ndarray(1, shape, strides, dtype);
}

STATIC uint8_t ndarray_init_helper(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE } },
        { MP_QSTR_dtype, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = NDARRAY_FLOAT } },
    };
    
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(1, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    
    uint8_t dtype = args[1].u_int;
    // at this point, dtype can still be `?` for Boolean arrays
    return dtype;
}

STATIC mp_obj_t ndarray_make_new_core(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args, mp_map_t *kw_args) {
    uint8_t dtype = ndarray_init_helper(n_args, args, kw_args);

    mp_obj_t len_in = mp_obj_len_maybe(args[0]);
	size_t i = 0, len1 = 0, len2 = 0;
    if (len_in == MP_OBJ_NULL) {
        mp_raise_ValueError(translate("first argument must be an iterable"));
    } else {
		// len1 is either the number of rows (for matrices), or the number of elements (row vectors)
		len1 = MP_OBJ_SMALL_INT_VALUE(len_in);
    }
    
    ndarray_obj_t *self;

	// TODO: this doesn't allow dtype conversion. 
    if(MP_OBJ_IS_TYPE(args[0], &ulab_ndarray_type)) {
        ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(args[0]);
        self = ndarray_copy_view(ndarray);
        return MP_OBJ_FROM_PTR(self);
    }
    
    // We have to figure out, whether the first element of the iterable is an iterable itself
    // Perhaps, there is a more elegant way of handling this
    mp_obj_iter_buf_t iter_buf1;
    mp_obj_t item1, iterable1 = mp_getiter(args[0], &iter_buf1);
    while ((item1 = mp_iternext(iterable1)) != MP_OBJ_STOP_ITERATION) {
        len_in = mp_obj_len_maybe(item1);
        if(len_in != MP_OBJ_NULL) { // indeed, this seems to be an iterable
            // Next, we have to check, whether all elements in the outer loop have the same length
            if(i > 0) {
                if(len2 != MP_OBJ_SMALL_INT_VALUE(len_in)) {
                    mp_raise_ValueError(translate("iterables are not of the same length"));
                }
            }
            len2 = MP_OBJ_SMALL_INT_VALUE(len_in);
            i++;
        }
    }
    // By this time, it should be established, what the shape is, so we can now create the array
    if(len2 == 0) {
		self = ndarray_new_linear_array(len1, dtype);
	} else {
		size_t shape[2] = {len1, len2};
		self = ndarray_new_dense_ndarray(2, shape, dtype);
	}
    
    size_t idx = 0;
    iterable1 = mp_getiter(args[0], &iter_buf1);
    if(len2 == 0) { // the first argument is a single iterable
        ndarray_assign_elements(self, iterable1, dtype, &idx);
    } else {
        mp_obj_iter_buf_t iter_buf2;
        mp_obj_t iterable2;
        while ((item1 = mp_iternext(iterable1)) != MP_OBJ_STOP_ITERATION) {
            iterable2 = mp_getiter(item1, &iter_buf2);
            ndarray_assign_elements(self, iterable2, dtype, &idx);
        }
    }
    return MP_OBJ_FROM_PTR(self);
}

#ifdef CIRCUITPY
mp_obj_t ndarray_make_new(const mp_obj_type_t *type, size_t n_args, const mp_obj_t *args, mp_map_t *kw_args) {
    mp_arg_check_num(n_args, kw_args, 1, 2, true);
    size_t n_kw = 0;
    if (kw_args != 0) {
        n_kw = kw_args->used;
    }
    mp_map_init_fixed_table(kw_args, n_kw, args + n_args);
    return ndarray_make_new_core(type, n_args, n_kw, args, kw_args);
}
#else
mp_obj_t ndarray_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 1, 2, true);
    mp_map_t kw_args;
    mp_map_init_fixed_table(&kw_args, n_kw, args + n_args);
    return ndarray_make_new_core(type, n_args, n_kw, args, &kw_args);
}
#endif

mp_bound_slice_t generate_slice(mp_uint_t n, mp_obj_t index) {
    mp_bound_slice_t slice;
    if(MP_OBJ_IS_TYPE(index, &mp_type_slice)) {
		mp_obj_slice_indices(index, n, &slice);
    } else if(mp_obj_is_int(index)) {
        int32_t _index = mp_obj_get_int(index);
        if(_index < 0) {
            _index += n;
        } 
        if((_index >= n) || (_index < 0)) {
            mp_raise_msg(&mp_type_IndexError, "index is out of bounds");
        }
        slice.start = _index;
        slice.stop = _index + 1;
        slice.step = 1;
    } else {
        mp_raise_msg(&mp_type_IndexError, "indices must be integers, slices, or Boolean lists");
    }
    return slice;
}

size_t slice_length(mp_bound_slice_t slice) {
    int32_t len, correction = 1;
    if(slice.step > 0) correction = -1;
    len = (slice.stop - slice.start + (slice.step + correction)) / slice.step;
    if(len < 0) return 0;
    return (size_t)len;
}

ndarray_obj_t *ndarray_new_view_from_tuple(ndarray_obj_t *ndarray, mp_obj_tuple_t *slices) {
	// generates a new view from a tuple of slices
	if(slices->len > ndarray->ndim) {
		mp_raise_msg(&mp_type_IndexError, "too many indices for array");
	}
	
    mp_bound_slice_t slice;
    size_t *shape_array = m_new(size_t, ndarray->ndim);
    int32_t *strides_array = m_new(int32_t, ndarray->ndim);
    size_t offset = 0;
    uint8_t ndim = 0;
    for(uint8_t i=0; i < ndarray->ndim; i++) {
        if(i < slices->len) {
            slice = generate_slice(ndarray->shape[i], slices->items[i]);
			offset += slice.start * ndarray->strides[i];
            size_t len = slice_length(slice);
			if(len == 0) { // we have ended up with an empty array, so we can return immediately
				shape_array[0] = 0;
				return ndarray_new_view(ndarray, 1, shape_array, strides_array, offset);
			}
            if(len == 1) { // this dimension is removed from the array, do nothing
			} else {
				shape_array[ndim] = slice_length(slice);
				strides_array[ndim] = ndarray->strides[i] * slice.step;
				ndim++;
			}
        } else {
            shape_array[ndim] = ndarray->shape[i];
            strides_array[ndim] = ndarray->strides[i];
            ndim++;
        }
    }
    if(ndim == 0) { 
		// this means that the slice length was always 1, i.e., we have to return a single scalar
		return mp_binary_get_val_array(ndarray->dtype, ndarray->array, offset);
	}
    return ndarray_new_view(ndarray, ndim, shape_array, strides_array, offset);
}

mp_obj_t ndarray_subscript_assign(ndarray_obj_t *lhs, ndarray_obj_t *rhs) {
	// assign the values in the right-hand-side array into the left-hand-side
	// array, and stretches the axes, if necessary
	if(rhs->ndim > lhs->ndim) {
		mp_raise_ValueError(translate("right hand side is out of bounds"));
	}
	size_t rshape[ULAB_MAX_DIMS];
	size_t lcoords[ULAB_MAX_DIMS];
	// we can only assign into a tensor, if the right hand side shape 
	// fulfils one of the conditions
	// 
	// - axis is missing (counted from the last dimension)
	// - axis length is equal to the axis length on the left hand side, 
	// - axis length is equal to 1
	uint8_t diff_ndim = lhs->ndim - rhs->ndim;
	for(uint8_t i=0; i < lhs->ndim; i++) {
		lcoords[i] = 0;
		if(diff_ndim > i) { // missing shape on RHS
			rshape[i] = 0;
		} else {
			if(lhs->shape[i] == rhs->shape[i-diff_ndim]) {
				// shape on RHS is equal to shape on LHS
				rshape[i] = 1;
			} else if(rhs->shape[i-diff_ndim] == 1) {
				// shape on RHS is simply 1
				rshape[i] = 0;
			} else {
				mp_raise_ValueError(translate("incompatible shapes in assignment"));
			}
		}
	}
	// work with mp_obj_t, so that we don't have to deal with dtypes
    mp_obj_t item;
	int32_t loffset = 0;
	int32_t roffset = 0;
    for(size_t i=0; i < lhs->len; i++) {
        item = mp_binary_get_val_array(rhs->dtype, rhs->array, roffset);
        mp_binary_set_val_array(lhs->dtype, lhs->array, loffset, item);
        loffset += lhs->strides[lhs->ndim-1];
		lcoords[lhs->ndim-1] += 1;
		roffset += rhs->strides[rhs->ndim-1] * rshape[lhs->ndim-1];
        for(uint8_t j=lhs->ndim-1; j > 0; j--) {
            if(lcoords[j] == lhs->shape[j]) { // we are at a dimension boundary
				loffset -= lhs->shape[j] * lhs->strides[j];
                loffset += lhs->strides[j-1];
				roffset -= rshape[j] * rhs->strides[j-diff_ndim];
				roffset += rshape[j-1] * rhs->strides[j-diff_ndim-1];
				lcoords[j] = 0;
                lcoords[j-1] += 1;
				if(j <= diff_ndim) roffset = 0;
            } else { // coordinates can change only, if the last coordinate changes
                break;
            }
        }
    }
    return MP_OBJ_FROM_PTR(lhs);
}

ndarray_obj_t *ndarray_from_mp_obj(mp_obj_t object) {
	// returns a rank-one ndarray, if the object is a scalar,
	// or the object itself, if it is an ndarray
	uint8_t dtype;
	if(MP_OBJ_IS_TYPE(object, &ulab_ndarray_type)) {
		return MP_OBJ_TO_PTR(object);
	} else if(MP_OBJ_IS_INT(object)) {
		int32_t ivalue = mp_obj_get_int(object);
		if((ivalue > 0) && (ivalue < 256)) {
			dtype = NDARRAY_UINT8;
		} else if((ivalue > 255) && (ivalue < 65535)) {
			dtype = NDARRAY_UINT16;
		} else if((ivalue < 0) && (ivalue > -128)) {
			dtype = NDARRAY_INT8;
		} else if((ivalue < -127) && (ivalue > -32767)) {
			dtype = NDARRAY_INT16;
		} else { // the integer value clearly does not fit the ulab integer types, so move on to float
			dtype = NDARRAY_FLOAT;
		}
    } else if(mp_obj_is_float(object)) {
		dtype = NDARRAY_FLOAT;
    } else {
        mp_raise_TypeError(translate("wrong operand type"));
    }
	ndarray_obj_t *ndarray = ndarray_new_linear_array(1, dtype);
	mp_binary_set_val_array(dtype, ndarray->array, 0, object);
	return ndarray;
}

mp_obj_t ndarray_subscr(mp_obj_t self_in, mp_obj_t index, mp_obj_t value) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    
    if (value == MP_OBJ_SENTINEL) { // return value(s)
		// turn all possibilities into a tuple of slices
		mp_obj_t tuple;
		mp_obj_t *items = m_new(mp_obj_t, 1);
        if(mp_obj_is_int(index)) {
			mp_bound_slice_t _index = generate_slice(self->shape[0], index);
			items[0] = mp_obj_new_slice(mp_obj_new_int(_index.start), mp_obj_new_int(_index.stop), mp_obj_new_int(_index.step));
			tuple = mp_obj_new_tuple(1, items);
			return MP_OBJ_FROM_PTR(ndarray_new_view_from_tuple(self, tuple));
		} else if(MP_OBJ_IS_TYPE(index, &mp_type_slice)) {
			items[0] = index;
			tuple = mp_obj_new_tuple(1, items);
			return MP_OBJ_FROM_PTR(ndarray_new_view_from_tuple(self, tuple));
		} else if(MP_OBJ_IS_TYPE(index, &mp_type_tuple)) {
			return MP_OBJ_FROM_PTR(ndarray_new_view_from_tuple(self, index));
		} else {
			mp_raise_ValueError(translate("wrong index type in array"));
		}			
    } else { // assignment to slices; the value must be an ndarray, or a scalar
		if(!MP_OBJ_IS_TYPE(value, &ulab_ndarray_type) && 
			!mp_obj_is_int(value) && !mp_obj_is_float(value)) {
			mp_raise_ValueError(translate("right hand side must be an ndarray, or a scalar"));
        } 
		mp_obj_t tuple;
		mp_obj_t *items = m_new(mp_obj_t, 1);
		ndarray_obj_t *lhs = NULL;
		if(mp_obj_is_int(index)) {
			mp_bound_slice_t _index = generate_slice(self->shape[0], index);
			items[0] = mp_obj_new_slice(mp_obj_new_int(_index.start), mp_obj_new_int(_index.stop), mp_obj_new_int(_index.step));
			tuple = mp_obj_new_tuple(1, items);
			lhs = ndarray_new_view_from_tuple(self, tuple);
		} else if(MP_OBJ_IS_TYPE(index, &mp_type_slice)) {
			items[0] = index;
			tuple = mp_obj_new_tuple(1, items);
			lhs = ndarray_new_view_from_tuple(self, tuple);
		} else if(MP_OBJ_IS_TYPE(index, &mp_type_tuple)) {
			lhs = ndarray_new_view_from_tuple(self, index);
		} else {
			mp_raise_ValueError(translate("wrong index type in array"));
		}
		// at this point, we have a view that is of the correct shape, 
		// so we can compare it to the value
		ndarray_obj_t *rhs;
		if(mp_obj_is_int(value) || mp_obj_is_float(value)) {
			rhs = ndarray_from_mp_obj(value);
		} else { // at this point, the right hand side can only be an ndarray, no need to check
			rhs = MP_OBJ_TO_PTR(value);
		}
		return ndarray_subscript_assign(lhs, rhs);
    }
    return mp_const_none;
}

// itarray iterator
// the iterator works in the general n-dimensional case
mp_obj_t ndarray_getiter(mp_obj_t o_in, mp_obj_iter_buf_t *iter_buf) {
    return mp_obj_new_ndarray_iterator(o_in, 0, iter_buf);
}

typedef struct _mp_obj_ndarray_it_t {
    mp_obj_base_t base;
    mp_fun_1_t iternext;
    mp_obj_t ndarray;
    size_t cur;
} mp_obj_ndarray_it_t;

mp_obj_t ndarray_iternext(mp_obj_t self_in) {
    mp_obj_ndarray_it_t *self = MP_OBJ_TO_PTR(self_in);
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(self->ndarray);
    size_t iter_end = ndarray->shape[0];
    if(self->cur < iter_end) {
        if(ndarray->ndim == 1) { // we have a linear array
            // read the current value
            self->cur++;
            return mp_binary_get_val_array(ndarray->dtype, ndarray->array, self->cur-1);
        } else { // we have a tensor, return the reduced view
            int32_t offset = self->cur * ndarray->strides[0];
			self->cur++;
            ndarray_obj_t *value = ndarray_new_view(ndarray, ndarray->ndim-1, ndarray->shape+1, ndarray->strides+1, offset);
            return MP_OBJ_FROM_PTR(value);
        }
    } else {
        return MP_OBJ_STOP_ITERATION;
    }
}

mp_obj_t mp_obj_new_ndarray_iterator(mp_obj_t ndarray, size_t cur, mp_obj_iter_buf_t *iter_buf) {
    assert(sizeof(mp_obj_ndarray_it_t) <= sizeof(mp_obj_iter_buf_t));
    mp_obj_ndarray_it_t *o = (mp_obj_ndarray_it_t*)iter_buf;
    o->base.type = &mp_type_polymorph_iter;
    o->iternext = ndarray_iternext;
    o->ndarray = ndarray;
    o->cur = cur;
    return MP_OBJ_FROM_PTR(o);
}
#if 0
broadcast_shape_t ndarray_can_broadcast(ndarray_obj_t *lhs, ndarray_obj_t *rhs) {
	// returns a structure with the three sets of strides that can be used 
	// in the iteration loop
    uint8_t ndim = MAX(lhs->ndim, rhs->ndim);
    uint8_t min_ndiff = MIN(lhs->ndim, rhs->ndim);
    uint8_t diff_ndim = lhs->ndim > rhs->ndim ? lhs->ndim - rhs->ndim : rhs->ndim - lhs->ndim;

	broadcast_shape_t result;
    if(ndim > ULAB_MAX_DIMS) {
		// do not deal with high-dimensional tensors for now
		result.broadcastable = false;
		return result;
	}
	
    for(uint8_t i=0; i < ndim; i++) {
		result->left_shape[i] = 0;
        result->right_shape[i] = 0;
        result->output_shape[i] = 0;
        result->left_strides[i] = 0;
        result->right_strides[i] = 0;
        result->output_strides[i] = 0;
    }
    result.broadcastable = true;
    for(uint8_t i=0; i < min_ndim; i++) {
        if((lhs->shape[lhs->ndim-i-1] == rhs->shape[rhs->ndim-i-1]) ||
           (lhs->shape[lhs->ndim-i-1] == 1) || (rhs->shape[rhs->ndim-i-1] == 1)) {
            result->target_shape[ndim-i-1] = lhs->shape[lhs->ndim-i-1] > rhs->shape[rhs->ndim-i-1] ? lhs->shape[lhs->ndim-i-1] : rhs->shape[rhs->ndim-i-1];
        } else {
            result->broadcastable = false;            
            break;
        }
    }
    if(result.broadcastable) {
		for(uint8_t i=0; i < min_ndim; i++) {
			if(lhs->shape[lhs->ndim-i-1] == 1) {
				if(rhs->shape[rhs->ndim-i-1] == 1)) {
					left_coords[ndim-1-i] = lhs->strides[lhs->ndim-i-1];
				}
			} else {
				left_coords[ndim-1-i] = 1;
			}
			if(rhs->shape[lhs->ndim-i-1] == 1) {
				if(lhs->shape[rhs->ndim-i-1] == 1)) {
					right_coords[ndim-1-i] = 1;
				}
			} else {
				right_coords[ndim-1-i] = rhs->strides[rhs->ndim-i-1];;
			}
		}
		if(left->ndim > right->ndim) {
			for(uint8_t i=0; i < diff) left_coords[i] = lhs->strides[i];
		} else {
			for(uint8_t i=0; i < diff) right_coords[i] = rhs->strides[i];
		}		
	}
    return result;
}

// Binary operations
mp_obj_t ndarray_binary_op(mp_binary_op_t op, mp_obj_t _lhs, mp_obj_t _rhs) {

    /*
    * Since the number of ndarray dtypes is lower than in numpy, we have to define
    * our own upcasting rules. These are stipulated here for now:
    * uint8 + int8 => int16
    * uint8 + int16 => int16
    * uint8 + uint16 => uint16
    * int8 + int16 => int16
    * int8 + uint16 => uint16
    * uint16 + int16 => float
    */

    // TODO: implement in-place operators    
    ndarray_obj_t *lhs = ndarray_from_binary_operand(_lhs);
    ndarray_obj_t *rhs = ndarray_from_binary_operand(_rhs);
    broadcast_shape_t bcast = ndarray_can_broadcast(lhs, rhs);
	if(!bcast.broadcastable) {
		mp_raise_ValueError(translate("operands could not be broadcast together"));
	}
	switch(op) {
		case MP_BINARY_OP_EQUAL:
			return mp_obj_new_int(lhs == rhs);
			break;
		case MP_BINARY_OP_LESS:
		case MP_BINARY_OP_LESS_EQUAL:
		case MP_BINARY_OP_MORE:
		case MP_BINARY_OP_MORE_EQUAL:
		case MP_BINARY_OP_ADD:
		case MP_BINARY_OP_SUBTRACT:
		case MP_BINARY_OP_TRUE_DIVIDE:
		case MP_BINARY_OP_MULTIPLY:
			// The parameters of RUN_BINARY_LOOP are 
			// dtype of result, type_out, type_left, type_right, lhs operand, rhs operand, operator
			if(lhs->dtype == NDARRAY_UINT8) {
				if(rhs->dtype == NDARRAY_UINT8) {
					RUN_BINARY_LOOP(NDARRAY_UINT8, uint8_t, uint8_t, uint8_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_INT8) {
					RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, uint8_t, int8_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_UINT16) {
					RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint8_t, uint16_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_INT16) {
					RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, uint8_t, int16_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_FLOAT) {
					RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint8_t, mp_float_t, lhs, rhs, op, bcast);
				}
			} else if(lhs->dtype == NDARRAY_INT8) {
				if(rhs->dtype == NDARRAY_UINT8) {
					RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, uint8_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_INT8) {
					RUN_BINARY_LOOP(NDARRAY_INT8, int8_t, int8_t, int8_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_UINT16) {
					RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, uint16_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_INT16) {
					RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, int16_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_FLOAT) {
					RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, int8_t, mp_float_t, lhs, rhs, op, bcast);
				}                
			} else if(lhs->dtype == NDARRAY_UINT16) {
				if(rhs->dtype == NDARRAY_UINT8) {
					RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, uint8_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_INT8) {
					RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, int8_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_UINT16) {
					RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, uint16_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_INT16) {
					RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint16_t, int16_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_FLOAT) {
					RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint8_t, mp_float_t, lhs, rhs, op, bcast);
				}
			} else if(lhs->dtype == NDARRAY_INT16) {
				if(rhs->dtype == NDARRAY_UINT8) {
					RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, uint8_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_INT8) {
					RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, int8_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_UINT16) {
					RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, int16_t, uint16_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_INT16) {
					RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, int16_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_FLOAT) {
					RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint16_t, mp_float_t, lhs, rhs, op, bcast);
				}
			} else if(lhs->dtype == NDARRAY_FLOAT) {
				if(rhs->dtype == NDARRAY_UINT8) {
					RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, uint8_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_INT8) {
					RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, int8_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_UINT16) {
					RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, uint16_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_INT16) {
					RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, int16_t, lhs, rhs, op, bcast);
				} else if(rhs->dtype == NDARRAY_FLOAT) {
					RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, mp_float_t, lhs, rhs, op, bcast);
				}
			} else { // this should never happen
				mp_raise_TypeError(translate("wrong input type"));
			}
			// this instruction should never be reached, but we have to make the compiler happy
			return MP_OBJ_NONE; 
		default:
			return MP_OBJ_NONE; // op not supported                                                        
	}
}
#endif
mp_obj_t ndarray_unary_op(mp_unary_op_t op, mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    uint8_t itemsize = mp_binary_get_size('@', self->dtype, NULL);
    ndarray_obj_t *ndarray = NULL;
    switch (op) {
        case MP_UNARY_OP_LEN: 
            if(self->ndim > 1) {
                return mp_obj_new_int(self->ndim);
            } else {
                return mp_obj_new_int(self->len);
            }
            break;
        
        case MP_UNARY_OP_INVERT:
            if(self->dtype == NDARRAY_FLOAT) {
                mp_raise_ValueError(translate("operation is not supported for given type"));
            }
            // we can invert the content byte by byte, no need to distinguish between different dtypes
            ndarray = ndarray_copy_view(self); // from this point, this is a dense copy
            uint8_t *array = (uint8_t *)ndarray->array;
            if(ndarray->boolean) {
                for(size_t i=0; i < ndarray->len; i++, array++) *array = *array ^ 0x01;
            } else {
                for(size_t i=0; i < ndarray->len*itemsize; i++, array++) *array ^= 0xFF;
            }
            return MP_OBJ_FROM_PTR(ndarray);
            break;
        
        case MP_UNARY_OP_NEGATIVE:
            ndarray = ndarray_copy_view(self); // from this point, this is a dense copy
            if(self->dtype == NDARRAY_UINT8) {
                uint8_t *array = (uint8_t *)ndarray->array;
                for(size_t i=0; i < self->len; i++, array++) *array = -(*array);
            } else if(self->dtype == NDARRAY_INT8) {
                int8_t *array = (int8_t *)ndarray->array;
                for(size_t i=0; i < self->len; i++, array++) *array = -(*array);
            } else if(self->dtype == NDARRAY_UINT16) {                
                uint16_t *array = (uint16_t *)ndarray->array;
                for(size_t i=0; i < self->len; i++, array++) *array = -(*array);
            } else if(self->dtype == NDARRAY_INT16) {
                int16_t *array = (int16_t *)ndarray->array;
                for(size_t i=0; i < self->len; i++, array++) *array = -(*array);
            } else {
                mp_float_t *array = (mp_float_t *)ndarray->array;
                for(size_t i=0; i < self->len; i++, array++) *array = -(*array);
            }
            return MP_OBJ_FROM_PTR(ndarray);
            break;

        case MP_UNARY_OP_POSITIVE:
            return MP_OBJ_FROM_PTR(ndarray_copy_view(self));

        case MP_UNARY_OP_ABS:
            ndarray = ndarray_copy_view(self);
            // if Booleam, NDARRAY_UINT8, or NDARRAY_UINT16, there is nothing to do
            if(self->dtype == NDARRAY_INT8) {
                int8_t *array = (int8_t *)ndarray->array;
                for(size_t i=0; i < self->len; i++, array++) {
                    if(*array < 0) *array = -(*array);
                }
            } else if(self->dtype == NDARRAY_INT16) {
                int16_t *array = (int16_t *)ndarray->array;
                for(size_t i=0; i < self->len; i++, array++) {
                    if(*array < 0) *array = -(*array);
                }
            } else {
                mp_float_t *array = (mp_float_t *)ndarray->array;
                for(size_t i=0; i < self->len; i++, array++) {
                    if(*array < 0) *array = -(*array);
                }                
            }
            return MP_OBJ_FROM_PTR(ndarray);
            break;
        default: return MP_OBJ_NULL; // operator not supported
    }
}

mp_obj_t ndarray_shape(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    mp_obj_t *items = m_new(mp_obj_t, self->ndim);
    for(uint8_t i=0; i < self->ndim; i++) {
        items[i] = mp_obj_new_int(self->shape[i]);
    }
    mp_obj_t tuple = mp_obj_new_tuple(self->ndim, items);
    m_del(mp_obj_t, items, self->ndim);
    return tuple;
}

MP_DEFINE_CONST_FUN_OBJ_1(ndarray_shape_obj, ndarray_shape);

mp_obj_t ndarray_strides(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    mp_obj_t *items = m_new(mp_obj_t, self->ndim);
    uint8_t itemsize = mp_binary_get_size('@', self->dtype, NULL);
    for(int8_t i=0; i < self->ndim; i++) {
        items[i] = mp_obj_new_int(self->strides[i]*itemsize);
    }
    mp_obj_t tuple = mp_obj_new_tuple(self->ndim, items);
    m_del(mp_obj_t, items, self->ndim);
    return tuple;
}

MP_DEFINE_CONST_FUN_OBJ_1(ndarray_strides_obj, ndarray_strides);

mp_obj_t ndarray_size(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(self->len);
}

mp_obj_t ndarray_itemsize(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return MP_OBJ_NEW_SMALL_INT(mp_binary_get_size('@', self->dtype, NULL));
}

MP_DEFINE_CONST_FUN_OBJ_1(ndarray_itemsize_obj, ndarray_itemsize);

mp_obj_t ndarray_ndim(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(self->ndim);
}

MP_DEFINE_CONST_FUN_OBJ_1(ndarray_ndim_obj, ndarray_ndim);

mp_obj_t ndarray_transpose(mp_obj_t self_in) {
    // TODO: check, what happens to the offset here, if we have a view
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    size_t *shape = m_new(size_t, self->ndim);
    int32_t *strides = m_new(int32_t, self->ndim);
    for(uint8_t i=0; i < self->ndim; i++) {
        shape[i] = self->shape[self->ndim-1-i];
        strides[i] = self->strides[self->ndim-1-i];
    }
    // TODO: I am not sure ndarray_new_view is OK here...
    ndarray_obj_t *ndarray = ndarray_new_view(self, self->ndim, shape, strides, 0);
    return MP_OBJ_FROM_PTR(ndarray);
}

MP_DEFINE_CONST_FUN_OBJ_1(ndarray_transpose_obj, ndarray_transpose);

mp_obj_t ndarray_reshape(mp_obj_t oin, mp_obj_t _shape) {
    // TODO: not all reshaping operations can be realised via views! Only dense arrays for now
    // returns a new view with the specified shape
    ndarray_obj_t *ndarray_in = MP_OBJ_TO_PTR(oin);
	if(!MP_OBJ_IS_TYPE(_shape, &mp_type_tuple)) {
		mp_raise_TypeError(translate("shape must be a tuple"));
    }

    mp_obj_tuple_t *shape = MP_OBJ_TO_PTR(_shape);
    if(shape->len > ULAB_MAX_DIMS) {
        mp_raise_ValueError(translate("maximum number of dimensions is 4"));
	}
    size_t *new_shape = m_new(size_t, shape->len);
    size_t new_length = 1;
    for(uint8_t i=0; i < shape->len; i++) {
        new_shape[i] = mp_obj_get_int(shape->items[i]);
        new_length *= new_shape[i];
    }
    
    if(ndarray_in->len != new_length) {
        mp_raise_ValueError(translate("input and output shapes are not compatible"));
    }

	ndarray_obj_t *ndarray;
	if(ndarray_is_dense(ndarray_in)) {
		int32_t *new_strides = strides_from_shape(new_shape, shape->len);
		ndarray = ndarray_new_view(ndarray_in, shape->len, new_shape, new_strides, 0);
	} else {
		ndarray = ndarray_new_ndarray_from_tuple(shape, ndarray_in->dtype);
		ndarray_copy_array(ndarray_in, ndarray);
	}
    return MP_OBJ_FROM_PTR(ndarray);
}

MP_DEFINE_CONST_FUN_OBJ_2(ndarray_reshape_obj, ndarray_reshape);
/*
mp_int_t ndarray_get_buffer(mp_obj_t self_in, mp_buffer_info_t *bufinfo, mp_uint_t flags) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    // buffer_p.get_buffer() returns zero for success, while mp_get_buffer returns true for success
    return !mp_get_buffer(self->array, bufinfo, flags);
}
*/
