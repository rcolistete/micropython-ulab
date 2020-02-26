
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "py/runtime.h"
#include "py/binary.h"
#include "py/obj.h"
#include "py/objtuple.h"
#include "ndarray.h"
#include "ndarray_properties.h"

// This function is copied verbatim from objarray.c
STATIC mp_obj_array_t *array_new(char typecode, size_t n) {
    int typecode_size = mp_binary_get_size('@', typecode, NULL);
    mp_obj_array_t *o = m_new_obj(mp_obj_array_t);
    // this step could probably be skipped: we are never going to store a bytearray per se
    #if MICROPY_PY_BUILTINS_BYTEARRAY && MICROPY_PY_ARRAY
    o->base.type = (typecode == BYTEARRAY_TYPECODE) ? &mp_type_bytearray : &mp_type_array;
    #elif MICROPY_PY_BUILTINS_BYTEARRAY
    o->base.type = &mp_type_bytearray;
    #else
    o->base.type = &mp_type_array;
    #endif
    o->typecode = typecode;
    o->free = 0;
    o->len = n;
    o->items = m_new(byte, typecode_size * o->len);
    return o;
}

mp_float_t ndarray_get_float_value(void *data, uint8_t typecode, size_t index) {
    if(typecode == NDARRAY_UINT8) {
        return (mp_float_t)((uint8_t *)data)[index];
    } else if(typecode == NDARRAY_INT8) {
        return (mp_float_t)((int8_t *)data)[index];
    } else if(typecode == NDARRAY_UINT16) {
        return (mp_float_t)((uint16_t *)data)[index];
    } else if(typecode == NDARRAY_INT16) {
        return (mp_float_t)((int16_t *)data)[index];
    } else {
        return (mp_float_t)((mp_float_t *)data)[index];
    }
}

void fill_array_iterable(mp_float_t *array, mp_obj_t iterable) {
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

void ndarray_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind) {
    (void)kind;
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    size_t offset = self->offset;
    uint8_t print_extra = self->ndim;
    size_t *coords = ndarray_new_coords(self->ndim);
    int32_t last_stride = self->strides[self->ndim-1];
    
    mp_print_str(print, "array(");
        
    for(size_t i=0; i < self->len; i++) {
        for(uint8_t j=0; j < print_extra; j++) {
            mp_print_str(print, "[");
        }
        print_extra = 0;
        if(!self->boolean) {
            mp_obj_print_helper(print, mp_binary_get_val_array(self->array->typecode, self->array->items, offset), PRINT_REPR);
        } else {
            if(((uint8_t *)self->array->items)[offset]) {
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
            mp_print_str(print, ",\n");
            if(print_extra > 1) {
                mp_print_str(print, "\n");
            }
        }
    }
    mp_print_str(print, "]");
    m_del(size_t, coords, self->ndim);

    if(self->boolean) {
        mp_print_str(print, ", dtype=bool)");
    } else if(self->array->typecode == NDARRAY_UINT8) {
        mp_print_str(print, ", dtype=uint8)");
    } else if(self->array->typecode == NDARRAY_INT8) {
        mp_print_str(print, ", dtype=int8)");
    } else if(self->array->typecode == NDARRAY_UINT16) {
        mp_print_str(print, ", dtype=uint16)");
    } else if(self->array->typecode == NDARRAY_INT16) {
        mp_print_str(print, ", dtype=int16)");
    } else if(self->array->typecode == NDARRAY_FLOAT) {
        mp_print_str(print, ", dtype=float)");
    }
}

void ndarray_assign_elements(mp_obj_array_t *data, mp_obj_t iterable, uint8_t typecode, size_t *idx) {
    // assigns a single row in the matrix
    mp_obj_t item;
    while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
        mp_binary_set_val_array(typecode, data->items, (*idx)++, item);
    }
}

ndarray_obj_t *ndarray_new_ndarray(uint8_t ndim, size_t *shape, int32_t *strides, uint8_t typecode) {
    // Creates the base ndarray with shape, and initialises the values to straight 0s
    // the function should work in the general n-dimensional case
    ndarray_obj_t *ndarray = m_new_obj(ndarray_obj_t);
    ndarray->base.type = &ulab_ndarray_type;
    ndarray->ndim = ndim;
    ndarray->shape = shape;
    ndarray->strides = strides;
    ndarray->offset = 0;
    if(typecode == NDARRAY_BOOL) {
        typecode = NDARRAY_UINT8;
        ndarray->boolean = NDARRAY_BOOLEAN;
    } else {
        ndarray->boolean = NDARRAY_NUMERIC;
    }
    ndarray->len = 1;
    for(uint8_t i=0; i < ndim; i++) {
        ndarray->len *= shape[i];
    }
    mp_obj_array_t *array = array_new(typecode, ndarray->len);
    // this should set all elements to 0, irrespective of the of the typecode (all bits are zero)
    // we could, perhaps, leave this step out, and initialise the array only, when needed
    memset(array->items, 0, array->len); 
    ndarray->array = array;
    return ndarray;
}

ndarray_obj_t *ndarray_new_dense_ndarray(uint8_t ndim, size_t *shape, uint8_t typecode) {
    // creates a dense array, i.e., one, where the strides are derived directly from the shapes
    // the function should work in the general n-dimensional case
    int32_t *strides = m_new(int32_t, ndim);
    strides[ndim-1] = 1;
    for(size_t i=ndim-1; i > 0; i--) {
        strides[i-1] = strides[i] * shape[i-1];
    }
    return ndarray_new_ndarray(ndim, shape, strides, typecode);
}

ndarray_obj_t *ndarray_new_view(mp_obj_array_t *array, uint8_t ndim, size_t *shape, int32_t *strides, size_t offset, uint8_t boolean) {
    // creates a new view from the input arguments
    // the function should work in the n-dimensional case
    ndarray_obj_t *ndarray = m_new_obj(ndarray_obj_t);
    ndarray->base.type = &ulab_ndarray_type;
    ndarray->boolean = boolean;
    ndarray->ndim = ndim;
    ndarray->shape = shape;
    ndarray->strides = strides;
    ndarray->len = 1;
    for(uint8_t i=0; i < ndim; i++) {
        ndarray->len *= shape[i];
    }    
    ndarray->offset = offset;
    ndarray->array = array;
    return ndarray;
}

ndarray_obj_t *ndarray_copy_view(ndarray_obj_t *source) {
    // creates a one-to-one deep copy of the input ndarray or its view
    // the function should work in the general n-dimensional case
    // In order to make it dtype-agnostic, we copy the memory content 
    // instead of reading out the values
    int32_t *strides = strides_from_shape(source->shape, source->ndim);

    uint8_t typecode = source->array->typecode;
    if(source->boolean) {
        typecode = NDARRAY_BOOLEAN;
    }
    ndarray_obj_t *ndarray = ndarray_new_ndarray(source->ndim, source->shape, strides, typecode);

    size_t *coords = ndarray_new_coords(source->ndim);
    int32_t last_stride = source->strides[source->ndim-1];
    uint8_t itemsize = mp_binary_get_size('@', source->array->typecode, NULL);

    uint8_t *array = (uint8_t *)source->array->items;
    array += source->offset * itemsize;
    uint8_t *new_array = (uint8_t *)ndarray->array->items;

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
        // TOOD: I haven't the foggiest idea as to why mp_const_none_obj causes problems.
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
    if (len_in == MP_OBJ_NULL) {
        mp_raise_ValueError(translate("first argument must be an iterable"));
    } 
    
    size_t len = MP_OBJ_SMALL_INT_VALUE(len_in);
    ndarray_obj_t *self, *ndarray;

    if(MP_OBJ_IS_TYPE(args[0], &ulab_ndarray_type)) {
        ndarray = MP_OBJ_TO_PTR(args[0]);
        self = ndarray_copy_view(ndarray);
        return MP_OBJ_FROM_PTR(self);
    }
    // work with a single dimension for now
    self = ndarray_new_linear_array(len, dtype);
    
    size_t i = 0;
    mp_obj_iter_buf_t iter_buf;
    mp_obj_t item, iterable = mp_getiter(args[0], &iter_buf);
    if(self->boolean) {
        uint8_t *array = (uint8_t *)self->array->items;
        while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
            // TODO: I think this is wrong here: we have to check for the trueness of item
            if(mp_obj_get_float(item)) {
                *array = 1;
            }
            array++;
        }
    } else {
        while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
            mp_binary_set_val_array(dtype, self->array->items, i++, item);
        }
    }
/*
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
    ndarray_obj_t *self = create_new_ndarray((len2 == 0) ? 1 : len1, (len2 == 0) ? len1 : len2, dtype);
    iterable1 = mp_getiter(args[0], &iter_buf1);
    i = 0;
    if(len2 == 0) { // the first argument is a single iterable
        ndarray_assign_elements(self->array, iterable1, dtype, &i);
    } else {
        mp_obj_iter_buf_t iter_buf2;
        mp_obj_t iterable2; 

        while ((item1 = mp_iternext(iterable1)) != MP_OBJ_STOP_ITERATION) {
            iterable2 = mp_getiter(item1, &iter_buf2);
            ndarray_assign_elements(self->array, iterable2, dtype, &i);
        }
    }
    */
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
    // micropython seems to have difficulties with negative steps
    mp_bound_slice_t slice;
    if(MP_OBJ_IS_TYPE(index, &mp_type_slice)) {
        mp_seq_get_fast_slice_indexes(n, index, &slice);
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
    mp_bound_slice_t slice;
    size_t *shape_array = m_new(size_t, ndarray->ndim);
    int32_t *strides_array = m_new(int32_t, ndarray->ndim);
    size_t offset = ndarray->offset;
    for(uint8_t i=0; i < ndarray->ndim; i++) {
        if(i < slices->len) {
            slice = generate_slice(ndarray->shape[i], slices->items[i]);
            offset += ndarray->offset + slice.start * ndarray->strides[i];
            shape_array[i] = slice_length(slice);
            strides_array[i] = ndarray->strides[i] * slice.step;
        } else {
            shape_array[i] = ndarray->shape[i];
            strides_array[i] = ndarray->strides[i]; 
        }
    }
    return ndarray_new_view(ndarray->array, ndarray->ndim, shape_array, strides_array, offset, ndarray->boolean);
}
    
bool ndarray_check_compatibility(ndarray_obj_t *lhs, ndarray_obj_t *rhs) {
    if(rhs->ndim > lhs->ndim) {
        return false;
    } 
    for(uint8_t i=0; i < rhs->ndim; i++) {
        if((rhs->shape[rhs->ndim-1-i] != 1) && (rhs->shape[rhs->ndim-1-i] != lhs->shape[lhs->ndim-1-i])) {
            return false;
        }
    }
    return true;
}

mp_obj_t ndarray_assign_view_from_tuple(ndarray_obj_t *ndarray, mp_obj_tuple_t *slices, mp_obj_t value) {
    ndarray_obj_t *lhs = ndarray_new_view_from_tuple(ndarray, slices);
    ndarray_obj_t *rhs;
    if(MP_OBJ_IS_TYPE(value, &ulab_ndarray_type)) {
        rhs = MP_OBJ_TO_PTR(value);
        // since this is an assignment, the left hand side should definitely be able to contain the right hand side
        if(!ndarray_check_compatibility(lhs, rhs)) {
            mp_raise_ValueError("could not broadcast input array into output array");
        }
    } else { // we have a scalar, so create an ndarray for it
        size_t *shape = m_new(size_t, lhs->ndim*sizeof(size_t));
        for(uint8_t i=0; i < lhs->ndim; i++) {
            shape[i] = 1;
        }
        rhs = ndarray_new_dense_ndarray(lhs->ndim, shape, lhs->array->typecode);
        mp_binary_set_val_array(rhs->array->typecode, rhs->array->items, 0, value);
    }
    size_t roffset = rhs->offset;
    size_t loffset = lhs->offset;
    size_t *lcoords = ndarray_new_coords(lhs->ndim);
    mp_obj_t item;
    uint8_t diff_ndim = lhs->ndim - rhs->ndim;
    for(size_t i=0; i < lhs->len; i++) {
        item = mp_binary_get_val_array(rhs->array->typecode, rhs->array->items, roffset);
        mp_binary_set_val_array(lhs->array->typecode, lhs->array->items, loffset, item);
        for(uint8_t j=lhs->ndim-1; j > 0; j--) {
            loffset += lhs->strides[j];
            lcoords[j] += 1;
            if(j >= diff_ndim) {
                if(rhs->shape[j-diff_ndim] != 1) {
                    roffset += rhs->strides[j-diff_ndim];
                }
            }
            if(lcoords[j] == lhs->shape[j]) { // we are at a dimension boundary
                if(j > diff_ndim) { // this means right-hand-side coordinates that haven't been prepended
                    if(rhs->shape[j-diff_ndim] != 1) {
                        // if rhs->shape[j-diff_ndim] != 1, then rhs->shape[j-diff_ndim] == lhs->shape[j],
                        // so we can advance the offset counter
                        roffset -= rhs->shape[j-diff_ndim] * rhs->strides[j-diff_ndim];
                        roffset += rhs->strides[j-diff_ndim-1];
                    } else { // rhs->shape[j-diff_ndim] == 1
                        roffset -= rhs->strides[j-diff_ndim];
                    }
                } else {
                    roffset = rhs->offset;
                }
                loffset -= lhs->shape[j] * lhs->strides[j];
                loffset += lhs->strides[j-1];
                lcoords[j] = 0;
                lcoords[j-1] += 1;
            } else { // coordinates can change only, if the last coordinate changes
                break;
            }
        }
    }
    m_del(size_t, lcoords, lhs->ndim);
    return mp_const_none;
}

mp_obj_t ndarray_subscr(mp_obj_t self_in, mp_obj_t index, mp_obj_t value) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    
    if (value == MP_OBJ_SENTINEL) { // return value(s)
        if(mp_obj_is_int(index) || MP_OBJ_IS_TYPE(index, &mp_type_slice)) {
            mp_obj_t *items = m_new(mp_obj_t, 1);
            items[0] = index;
            mp_obj_t tuple = mp_obj_new_tuple(1, items);
            return ndarray_new_view_from_tuple(self, tuple);
        }                          
    } else { // assignment to slices; the value must be an ndarray, or a scalar
        return mp_const_none;
        #if 0
        if(!MP_OBJ_IS_TYPE(value, &ulab_ndarray_type) && 
          !MP_OBJ_IS_INT(value) && !mp_obj_is_float(value)) {
            mp_raise_ValueError(translate("right hand side must be an ndarray, or a scalar"));
        } else {
            ndarray_obj_t *values = NULL;
            if(MP_OBJ_IS_INT(value)) {
                values = create_new_ndarray(1, 1, self->array->typecode);
                mp_binary_set_val_array(values->array->typecode, values->array->items, 0, value);   
            } else if(mp_obj_is_float(value)) {
                values = create_new_ndarray(1, 1, NDARRAY_FLOAT);
                mp_binary_set_val_array(NDARRAY_FLOAT, values->array->items, 0, value);
            } else {
                values = MP_OBJ_TO_PTR(value);
            }
            return ndarray_get_slice(self, index, values);
        }
        #endif
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
            return mp_binary_get_val_array(ndarray->array->typecode, ndarray->array->items, self->cur-1);
        } else { // we have a tensor, return the reduced view
            size_t offset = ndarray->offset + self->cur * ndarray->strides[0];
            ndarray_obj_t *value = ndarray_new_view(ndarray->array, ndarray->ndim-1, ndarray->shape+1, ndarray->strides+1, offset, ndarray->boolean);
            self->cur++;
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

size_t *can_broadcast(ndarray_obj_t *lhs, ndarray_obj_t *rhs) {
    // returns the NULL, if arrays cannot be broadcast together, 
    // the new shape array otherwise
    uint8_t ndim = MAX(lhs->ndim, rhs->ndim);
    uint8_t diff = lhs->ndim > rhs->ndim ? lhs->ndim - rhs->ndim : rhs->ndim - lhs->ndim;
    
    size_t *shape = m_new(size_t, ndim);
    for(uint8_t i=0; i < ndim; i++) {
        shape[i] = 1;
    }
    
    for(uint8_t i=0; i < ndim-diff; i++) {
        if((lhs->shape[lhs->ndim-i-1] == rhs->shape[rhs->ndim-i-1]) ||
           (lhs->shape[lhs->ndim-i-1] == 1) || (rhs->shape[rhs->ndim-i-1] == 1)) {
            shape[ndim-i-1] = lhs->shape[lhs->ndim-i-1] > rhs->shape[rhs->ndim-i-1] ? lhs->shape[lhs->ndim-i-1] : rhs->shape[rhs->ndim-i-1];
        } else {
            m_del(size_t, shape, ndim);
            shape = NULL;
            break;
        }
    }
    return shape;
}

#define GET_FLOAT_VALUE(type, in, out) do {\
    if((type) == 1) (out) = (float)(*(unsigned char *)(in));\
    else if((type) == 2) (out) = (float)(*(signed char *)(in));\
    else if((type) == 3) (out) = (float)(*(unsigned int *)(in));\
    else if((type) == 4) (out) = (float)(*(signed int *)(in));\
    else (out) = (float)(*(float *)(in));\
} while(0)

// Binary operations
mp_obj_t ndarray_binary_op_helper(mp_binary_op_t op, mp_obj_t _lhs, mp_obj_t _rhs) {
    /*
    * Since the number of ndarray types is lower than in numpy, we have to define
    * our own upcasting rules. These are stipulated here for now:
    * uint8 + int8 => int16
    * uint8 + int16 => int16
    * uint8 + uint16 => uint16
    * int8 + int16 => int16
    * int8 + uint16 => uint16
    * uint16 + int16 => float
    */
    
    // One of the operands is a scalar 
    // TODO: implement in-place operators
    mp_obj_t RHS = MP_OBJ_NULL;
    bool rhs_is_scalar = true;
    if(MP_OBJ_IS_INT(_rhs)) {
        int32_t ivalue = mp_obj_get_int(_rhs);
        if((ivalue > 0) && (ivalue < 256)) {
            CREATE_SINGLE_ITEM(RHS, uint8_t, NDARRAY_UINT8, ivalue);
        } else if((ivalue > 255) && (ivalue < 65535)) {
            CREATE_SINGLE_ITEM(RHS, uint16_t, NDARRAY_UINT16, ivalue);
        } else if((ivalue < 0) && (ivalue > -128)) {
            CREATE_SINGLE_ITEM(RHS, int8_t, NDARRAY_INT8, ivalue);
        } else if((ivalue < -127) && (ivalue > -32767)) {
            CREATE_SINGLE_ITEM(RHS, int16_t, NDARRAY_INT16, ivalue);
        } else { // the integer value clearly does not fit the ulab integer types, so move on to float
            CREATE_SINGLE_ITEM(RHS, mp_float_t, NDARRAY_FLOAT, ivalue);
        }
    } else if(mp_obj_is_float(_rhs)) {
        mp_float_t fvalue = mp_obj_get_float(_rhs);
        CREATE_SINGLE_ITEM(RHS, mp_float_t, NDARRAY_FLOAT, fvalue);
    } else {
        RHS = _rhs;
        rhs_is_scalar = false;
    }
    //else 
    if(MP_OBJ_IS_TYPE(_lhs, &ulab_ndarray_type) && MP_OBJ_IS_TYPE(RHS, &ulab_ndarray_type)) { 
        // next, the ndarray stuff
        ndarray_obj_t *lhs = MP_OBJ_TO_PTR(_lhs);
        ndarray_obj_t *rhs = MP_OBJ_TO_PTR(RHS);
        if(!rhs_is_scalar) {
            if(!can_broadcast(lhs, rhs)) {
                mp_raise_ValueError(translate("operands could not be broadcast together"));
            }
        }
        switch(op) {
            case MP_BINARY_OP_EQUAL:
                return mp_obj_new_int(lhs == rhs);
            /*
                // Two arrays are equal, if their shape, strides, typecode, and elements are equal
                // TODO: check, whether this is what numpy assumes
                if((ol->ndim != or->ndim) || (ol->len != or->len) || (ol->array->typecode != or->array->typecode)) {
                    return mp_const_false;
                } else {
                    for(uint8_t i=0; i < ol->ndim; i++) {
                        if((ol->shape[i] != or->shape[i]) || (ol->strides[i] != or->strides[i])) {
                            return mp_const_false;
                        }
                    }
                    size_t i = ol->len * mp_binary_get_size('@', ol->array->typecode, NULL);
                    uint8_t *l = (uint8_t *)ol->array->items;
                    uint8_t *r = (uint8_t *)or->array->items;
                    while(i) { // At this point, we can simply compare the bytes, the type is irrelevant
                        if(*l++ != *r++) {
                            return mp_const_false;
                        }
                        i--;
                    }
                    return mp_const_true;
                } */
                break;
            case MP_BINARY_OP_LESS:
            case MP_BINARY_OP_LESS_EQUAL:
            case MP_BINARY_OP_MORE:
            case MP_BINARY_OP_MORE_EQUAL:
            case MP_BINARY_OP_ADD:
            case MP_BINARY_OP_SUBTRACT:
            case MP_BINARY_OP_TRUE_DIVIDE:
            case MP_BINARY_OP_MULTIPLY:
                // TODO: I believe, this part can be made significantly smaller (compiled size)
                // by doing only the typecasting in the large ifs, and moving the loops outside
                // These are the upcasting rules
                // float always becomes float
                // operation on identical types preserves type
                // The parameters of RUN_BINARY_LOOP are 
                // typecode of result, type_out, type_left, type_right, lhs operand, rhs operand, operator
                if(lhs->array->typecode == NDARRAY_UINT8) {
                    if(rhs->array->typecode == NDARRAY_UINT8) {
                        RUN_BINARY_LOOP(NDARRAY_UINT8, uint8_t, uint8_t, uint8_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_INT8) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, uint8_t, int8_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_UINT16) {
                        RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint8_t, uint16_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_INT16) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, uint8_t, int16_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint8_t, mp_float_t, lhs, rhs, op);
                    }
                } else if(lhs->array->typecode == NDARRAY_INT8) {
                    if(rhs->array->typecode == NDARRAY_UINT8) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, uint8_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_INT8) {
                        RUN_BINARY_LOOP(NDARRAY_INT8, int8_t, int8_t, int8_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_UINT16) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, uint16_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_INT16) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, int16_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, int8_t, mp_float_t, lhs, rhs, op);
                    }                
                } else if(lhs->array->typecode == NDARRAY_UINT16) {
                    if(rhs->array->typecode == NDARRAY_UINT8) {
                        RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, uint8_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_INT8) {
                        RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, int8_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_UINT16) {
                        RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, uint16_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_INT16) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint16_t, int16_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint8_t, mp_float_t, lhs, rhs, op);
                    }
                } else if(lhs->array->typecode == NDARRAY_INT16) {
                    if(rhs->array->typecode == NDARRAY_UINT8) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, uint8_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_INT8) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, int8_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_UINT16) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, int16_t, uint16_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_INT16) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, int16_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint16_t, mp_float_t, lhs, rhs, op);
                    }
                } else if(lhs->array->typecode == NDARRAY_FLOAT) {
                    if(rhs->array->typecode == NDARRAY_UINT8) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, uint8_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_INT8) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, int8_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_UINT16) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, uint16_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_INT16) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, int16_t, lhs, rhs, op);
                    } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, mp_float_t, lhs, rhs, op);
                    }
                } else { // this should never happen
                    mp_raise_TypeError(translate("wrong input type"));
                }
                // this instruction should never be reached, but we have to make the compiler happy
                return MP_OBJ_NULL; 
            default:
                return MP_OBJ_NULL; // op not supported                                                        
        }
    } else {
        mp_raise_TypeError(translate("wrong operand type on the right hand side"));
    }
}

mp_obj_t ndarray_binary_op(mp_binary_op_t op, mp_obj_t lhs, mp_obj_t rhs) {
    // Swap operands, if the left hand side is a scalar
    if(!MP_OBJ_IS_TYPE(lhs, &ulab_ndarray_type)) {
        return ndarray_binary_op_helper(op, rhs, lhs);
    }
    return ndarray_binary_op_helper(op, lhs, rhs);
}

mp_obj_t ndarray_unary_op(mp_unary_op_t op, mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    uint8_t itemsize = mp_binary_get_size('@', self->array->typecode, NULL);
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
            if(self->array->typecode == NDARRAY_FLOAT) {
                mp_raise_ValueError(translate("operation is not supported for given type"));
            }
            // we can invert the content byte by byte, no need to distinguish between different typecodes
            ndarray = ndarray_copy_view(self); // from this point, this is a dense copy
            uint8_t *array = (uint8_t *)ndarray->array->items;
            if(ndarray->boolean) {
                for(size_t i=0; i < ndarray->len; i++, array++) *array = *array ^ 0x01;
            } else {
                for(size_t i=0; i < ndarray->len*itemsize; i++, array++) *array ^= 0xFF;
            }
            return MP_OBJ_FROM_PTR(ndarray);
            break;
        
        case MP_UNARY_OP_NEGATIVE:
            ndarray = ndarray_copy_view(self); // from this point, this is a dense copy
            if(self->array->typecode == NDARRAY_UINT8) {
                uint8_t *array = (uint8_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++, array++) *array = -(*array);
            } else if(self->array->typecode == NDARRAY_INT8) {
                int8_t *array = (int8_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++, array++) *array = -(*array);
            } else if(self->array->typecode == NDARRAY_UINT16) {                
                uint16_t *array = (uint16_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++, array++) *array = -(*array);
            } else if(self->array->typecode == NDARRAY_INT16) {
                int16_t *array = (int16_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++, array++) *array = -(*array);
            } else {
                mp_float_t *array = (mp_float_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++, array++) *array = -(*array);
            }
            return MP_OBJ_FROM_PTR(ndarray);
            break;

        case MP_UNARY_OP_POSITIVE:
            return MP_OBJ_FROM_PTR(ndarray_copy_view(self));

        case MP_UNARY_OP_ABS:
            ndarray = ndarray_copy_view(self);
            // if Booleam, NDARRAY_UINT8, or NDARRAY_UINT16, there is nothing to do
            if(self->array->typecode == NDARRAY_INT8) {
                int8_t *array = (int8_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++, array++) {
                    if(*array < 0) *array = -(*array);
                }
            } else if(self->array->typecode == NDARRAY_INT16) {
                int16_t *array = (int16_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++, array++) {
                    if(*array < 0) *array = -(*array);
                }
            } else {
                mp_float_t *array = (mp_float_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++, array++) {
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
    size_t *shape = (size_t *)self->shape;
    for(uint8_t i=0; i < self->ndim; i++) {
        items[i] = mp_obj_new_int(shape[i]);
    }
    mp_obj_t tuple = mp_obj_new_tuple(self->ndim, items);
    m_del(mp_obj_t, items, self->ndim);
    return tuple;
}

MP_DEFINE_CONST_FUN_OBJ_1(ndarray_shape_obj, ndarray_shape);

mp_obj_t ndarray_strides(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    mp_obj_t *items = m_new(mp_obj_t, self->ndim);
    int32_t *strides = (int32_t *)self->strides;
    for(int8_t i=0; i < self->ndim; i++) {
        items[i] = mp_obj_new_int(strides[i]);
    }
    mp_obj_t tuple = mp_obj_new_tuple(self->ndim, items);
    m_del(mp_obj_t, items, self->ndim);
    return tuple;
}

mp_obj_t ndarray_size(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(self->len);
}

mp_obj_t ndarray_itemsize(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return MP_OBJ_NEW_SMALL_INT(mp_binary_get_size('@', self->array->typecode, NULL));
}

mp_obj_t ndarray_ndim(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(self->ndim);
}

mp_obj_t ndarray_transpose(mp_obj_t self_in) {
    // this function should work with the n-dimensional case, too
    // TODO: check, what happens to the offset here!
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    size_t *shape = m_new(size_t, self->ndim);
    int32_t *strides = m_new(int32_t, self->ndim);
    for(uint8_t i=0; i < self->ndim; i++) {
        shape[i] = self->shape[self->ndim-1-i];
        strides[i] = self->strides[self->ndim-1-i];
    }
    // TODO: I am not sure ndarray_new_view is OK here...
    ndarray_obj_t *ndarray = ndarray_new_view(self->array, self->ndim, shape, strides, self->offset, self->boolean);
    return MP_OBJ_FROM_PTR(ndarray);
}

MP_DEFINE_CONST_FUN_OBJ_1(ndarray_transpose_obj, ndarray_transpose);

mp_obj_t ndarray_reshape(mp_obj_t oin, mp_obj_t _shape) {
    // TODO: not all reshaping operations can be realised via views! Only dense arrays for now
    // returns a new view with the specified shape
    ndarray_obj_t *ndarray_in = MP_OBJ_TO_PTR(oin);
    mp_obj_tuple_t *shape = MP_OBJ_TO_PTR(_shape);
    
    size_t *new_shape = m_new(size_t, shape->len);
    size_t new_length = 1;
    for(uint8_t i=0; i < shape->len; i++) {
        new_shape[i] = mp_obj_get_int(shape->items[i]);
        new_length *= new_shape[i];
    }
    
    if(ndarray_in->len != new_length) {
        mp_raise_ValueError("input and output shapes are not compatible");
    }

    int32_t *new_strides = strides_from_shape(new_shape, shape->len);

    // TODO: offset is 0 only for dense arrays!!!
    ndarray_obj_t *ndarray = ndarray_new_view(ndarray_in->array, shape->len, new_shape, new_strides, 0, ndarray_in->boolean);
    return MP_OBJ_FROM_PTR(ndarray);
}

MP_DEFINE_CONST_FUN_OBJ_2(ndarray_reshape_obj, ndarray_reshape);

mp_int_t ndarray_get_buffer(mp_obj_t self_in, mp_buffer_info_t *bufinfo, mp_uint_t flags) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    // buffer_p.get_buffer() returns zero for success, while mp_get_buffer returns true for success
    return !mp_get_buffer(self->array, bufinfo, flags);
}
