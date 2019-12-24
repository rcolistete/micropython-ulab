/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 Zoltán Vörös
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

// helper functions
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

size_t ndarray_index_from_contracted(size_t index, ndarray_obj_t *ndarray, int32_t *strides, uint8_t ndim, uint8_t axis) {
    // calculates the index in the original (linear) array from the index in the contracted (linear) array
    size_t q, new_index = 0;
    for(size_t i=0; i <= ndim-1; i++) {
        q = index / strides[i];
        if(i < axis) { 
            new_index += q * ndarray->strides[i];
        } else {
            new_index += q * ndarray->strides[i+1];            
        }
        index -= q * strides[i];
    }
    return new_index + ndarray->offset;
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
        offset += self->strides[self->ndim-1];
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

ndarray_obj_t *ndarray_new_ndarray(uint8_t ndim, size_t *shape, int32_t *strides, uint8_t typecode) {
    // Creates the base ndarray with shape, and initialises the values to straight 0s
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
    int32_t *strides = m_new(int32_t, ndim);
    strides[ndim-1] = 1;
    for(size_t i=ndim-1; i > 0; i--) {
        strides[i-1] = strides[i] * shape[i];
    }
    return ndarray_new_ndarray(ndim, shape, strides, typecode);
}

ndarray_obj_t *ndarray_new_view(mp_obj_array_t *array, uint8_t ndim, size_t *shape, int32_t *strides, size_t offset, uint8_t boolean) {
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

ndarray_obj_t *ndarray_copy_view(ndarray_obj_t *input, uint8_t typecode) {
    // Creates a new ndarray from the input
    // If the input was a sliced view, the output will inherit the shape, but not the strides

    int32_t *strides = m_new(int32_t, input->ndim);
    strides[input->ndim-1] = 1;
    for(uint8_t i=input->ndim-1; i > 0; i--) {
        strides[i-1] = strides[i] * input->shape[i];
    }
    ndarray_obj_t *ndarray = ndarray_new_ndarray(input->ndim, input->shape, strides, typecode);
    ndarray->boolean = input->boolean;
    
    mp_obj_t item;
    size_t offset = input->offset;
    size_t *coords = ndarray_new_coords(input->ndim);
    
    for(size_t i=0; i < ndarray->len; i++) {
        item = mp_binary_get_val_array(input->array->typecode, input->array->items, offset);
        mp_binary_set_val_array(typecode, ndarray->array->items, i, item);
        offset += input->strides[input->ndim-1];
        coords[input->ndim-1] += 1;
        for(uint8_t j=ndarray->ndim-1; j > 0; j--) {
            if(coords[j] == input->shape[j]) {
                offset -= input->shape[j] * input->strides[j];
                offset += input->strides[j-1];
                coords[j] = 0;
                coords[j-1] += 1;
            } else { // coordinates can change only, if the last coordinate changes
                break;
            }
        }
    }
    m_del(size_t, coords, input->ndim);
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
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_PTR(&mp_const_none_obj)} },
        { MP_QSTR_dtype, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = NDARRAY_FLOAT } },
    };
    
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(1, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    
    uint8_t dtype = args[1].u_int;
    // at this point, dtype can still be `?` for Boolean arrays
    return dtype;
}

mp_obj_t ndarray_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    // TODO: implement dtype, and copy keywords
    mp_arg_check_num(n_args, n_kw, 1, 2, true);
    mp_map_t kw_args;
    mp_map_init_fixed_table(&kw_args, n_kw, args + n_args);
    uint8_t dtype = ndarray_init_helper(n_args, args, &kw_args);

    mp_obj_t len_in = mp_obj_len_maybe(args[0]);
    size_t len = MP_OBJ_SMALL_INT_VALUE(len_in);
    ndarray_obj_t *self, *ndarray;
    
    if(MP_OBJ_IS_TYPE(args[0], &ulab_ndarray_type)) {
        ndarray = MP_OBJ_TO_PTR(args[0]);
        self = ndarray_copy_view(ndarray, dtype);
        return MP_OBJ_FROM_PTR(self);
    }
    // work with a single dimension for now
    self = ndarray_new_linear_array(len, dtype);
    
    size_t i=0;
    mp_obj_iter_buf_t iter_buf;
    mp_obj_t item, iterable = mp_getiter(args[0], &iter_buf);
    if(self->boolean) {
        uint8_t *array = (uint8_t *)self->array->items;
        while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
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
    return MP_OBJ_FROM_PTR(self);
}

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

mp_obj_t ndarray_subscr(mp_obj_t self, mp_obj_t index, mp_obj_t value) {
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(self);
    
    if (value == MP_OBJ_SENTINEL) { // return value(s)
        if(mp_obj_is_int(index) || MP_OBJ_IS_TYPE(index, &mp_type_slice)) {
            mp_obj_t *items = m_new(mp_obj_t, 1);
            items[0] = index;
            mp_obj_t tuple = mp_obj_new_tuple(1, items);
            return ndarray_new_view_from_tuple(ndarray, tuple);
        }
        // first, check, whether all members of the tuple are integer scalars, or slices
        if(MP_OBJ_IS_TYPE(index, &mp_type_tuple)) {
            mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(index);
            for(uint8_t i=0; i < tuple->len; i++) {
                if(!MP_OBJ_IS_TYPE(tuple->items[i], &mp_type_slice) && !mp_obj_is_int(tuple->items[i])) {
                    // TODO: we have to return a copy here
                    mp_raise_msg(&mp_type_IndexError, "wrong index type");
                }
            }
            // now we know that we can return a view
            ndarray_obj_t *result = ndarray_new_view_from_tuple(ndarray, tuple);
            return MP_OBJ_FROM_PTR(result);
        }
    } else { // assignment; the value must be an ndarray, or a scalar
        if(!MP_OBJ_IS_TYPE(value, &ulab_ndarray_type) && !mp_obj_is_int(value) && !mp_obj_is_float(value)) {
            mp_raise_ValueError("right hand side must be an ndarray, or a scalar");
        } else {
            if(mp_obj_is_int(index) || MP_OBJ_IS_TYPE(index, &mp_type_slice)) {
                mp_obj_t *items = m_new(mp_obj_t, 1);
                items[0] = index;
                mp_obj_t tuple = mp_obj_new_tuple(1, items);
                return ndarray_assign_view_from_tuple(ndarray, tuple, value);
            } 
            if(MP_OBJ_IS_TYPE(index, &mp_type_tuple)) {
                mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(index);
                for(uint8_t i=0; i < tuple->len; i++) {
                    if(!MP_OBJ_IS_TYPE(tuple->items[i], &mp_type_slice) && !mp_obj_is_int(tuple->items[i])) {
                        // TODO: we have to return a copy here
                        mp_raise_msg(&mp_type_IndexError, "wrong index type");
                    }
                }
                return ndarray_assign_view_from_tuple(ndarray, tuple, value);
            } else {
                mp_raise_NotImplementedError("wrong index type");
            }
        }
    }
    return mp_const_none;
}

// itarray iterator
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

mp_obj_t ndarray_info(mp_obj_t self_in) {
    // TODO: the proper way of handling this would be to use mp_print_str()
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    printf("class: ndarray\n");
    printf("shape: (%ld", (size_t)self->shape[0]);
    for(uint8_t i=1; i < self->ndim; i++) {
        printf(", %ld", (size_t)self->shape[i]);
    }
    printf(")");
    printf("\nstrides: (%ld", (size_t)self->strides[0]);
    for(uint8_t i=1; i < self->ndim; i++) {
        printf(", %ld", (size_t)self->strides[i]);
    }
    printf(")");
    printf("\nitemsize: %ld\n", mp_binary_get_size('@', self->array->typecode, NULL));
    printf("data pointer: %p\n", self->array->items);
    if(self->array->typecode == NDARRAY_BOOL) {
        printf("type: bool\n");
    } else if(self->array->typecode == NDARRAY_UINT8) {
        printf("type: uint8\n");
    } else if(self->array->typecode == NDARRAY_INT8) {
        printf("type: int8\n");
    } else if(self->array->typecode == NDARRAY_UINT16) {
        printf("type: uint16\n");
    } else if(self->array->typecode == NDARRAY_INT16) {
        printf("type: int16\n");
    } else {
        printf("type: float\n");
    }
    return mp_const_none;
}

mp_obj_t ndarray_flatten(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_order, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_QSTR(MP_QSTR_C)} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(pos_args[0]);
    
    GET_STR_DATA_LEN(args[0].u_obj, order, clen);
    if((clen != 1) || ((memcmp(order, "C", 1) != 0) && (memcmp(order, "F", 1) != 0))) {
        mp_raise_ValueError("flattening order must be either 'C', or 'F'");        
    }
    ndarray_obj_t *result = ndarray_new_linear_array(ndarray->len, ndarray->array->typecode);
    uint8_t itemsize = mp_binary_get_size('@', ndarray->array->typecode, NULL);
    if(ndarray->len == ndarray->array->len) { // this is a dense array, we can simply copy everything
        if(memcmp(order, "C", 1) == 0) { // C order; this should be fast, because no re-ordering is required
            memcpy(result->array->items, ndarray->array->items, itemsize*ndarray->len);
        } else { // Fortran order
            mp_raise_NotImplementedError("flatten is implemented in C order only");
        }
        return MP_OBJ_FROM_PTR(result);
    } else {
        uint8_t *rarray = (uint8_t *)result->array->items;
        uint8_t *narray = (uint8_t *)ndarray->array->items;
        size_t *coords = ndarray_new_coords(ndarray->ndim);

        size_t offset = ndarray->offset;
        if(memcmp(order, "C", 1) == 0) { // C order; this is a view, so we have to collect the items
            for(size_t i=0; i < result->len; i++) {
                memcpy(rarray, &narray[offset*itemsize], itemsize);
                rarray += itemsize;
                offset += ndarray->strides[ndarray->ndim-1];
                coords[ndarray->ndim-1] += 1;
                for(uint8_t j=ndarray->ndim-1; j > 0; j--) {
                    if(coords[j] == ndarray->shape[j]) {
                        offset -= ndarray->shape[j] * ndarray->strides[j];
                        offset += ndarray->strides[j-1];
                        coords[j] = 0;
                        coords[j-1] += 1;
                    } else { // coordinates can change only, if the last coordinate changes
                        break;
                    }
                }
            }
            m_del(size_t, coords, ndarray->ndim);
        } else { // Fortran order
            mp_raise_NotImplementedError("flatten is implemented for C order only");
        }
        //m_del(int32_t, shape_strides, 1);
        return MP_OBJ_FROM_PTR(result);
    }
}

mp_obj_t ndarray_reshape(mp_obj_t oin, mp_obj_t new_shape) {
    // TODO: not all reshaping operations can be realised via views!
    // returns a new view with the specified shape
    ndarray_obj_t *ndarray_in = MP_OBJ_TO_PTR(oin);
    mp_obj_tuple_t *shape = MP_OBJ_TO_PTR(new_shape);
    if(ndarray_in->len != ndarray_in->array->len) {
        mp_raise_ValueError("input and output shapes are not compatible");
    }
    size_t *shape_array = m_new(size_t, shape->len);
    int32_t *strides_array = m_new(int32_t, shape->len);
    size_t new_offset = ndarray_in->offset; // this has to be re-calculated
    strides_array[shape->len-1] = 1;
    shape_array[shape->len-1] = mp_obj_get_int(shape->items[shape->len-1]);
    for(uint8_t i=shape->len-1; i > 0; i--) {
        shape_array[i-1] = mp_obj_get_int(shape->items[i-1]);
        strides_array[i-1] = strides_array[i] * shape_array[i];
    }
    ndarray_obj_t *ndarray = ndarray_new_view(ndarray_in->array, shape->len, shape_array, strides_array, new_offset, ndarray_in->boolean);
    return MP_OBJ_FROM_PTR(ndarray);
}

mp_obj_t ndarray_transpose(mp_obj_t self_in) {
    // TODO: check, what happens to the offset here!
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    size_t *shape = m_new(size_t, self->ndim);
    int32_t *strides = m_new(int32_t, self->ndim);
    for(uint8_t i=0; i < self->ndim; i++) {
        shape[i] = self->shape[self->ndim-1-i];
        strides[i] = self->strides[self->ndim-1-i];
    }
    ndarray_obj_t *ndarray = ndarray_new_view(self->array, self->ndim, shape, strides, self->offset, self->boolean);
    return MP_OBJ_FROM_PTR(ndarray);
}

mp_obj_t ndarray_itemsize(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(mp_binary_get_size('@', self->array->typecode, NULL));
}

uint8_t upcasting(ndarray_obj_t *in1, ndarray_obj_t *in2) {
    if((in1->array->typecode = NDARRAY_FLOAT) || (in2->array->typecode = NDARRAY_FLOAT)) { // 9 cases
        return NDARRAY_FLOAT;
    } else if(in1->array->typecode == in2->array->typecode) { // 4 cases
        return in1->array->typecode;
    } else if(in1->array->typecode == NDARRAY_UINT8) { // 3 cases
        if(in2->array->typecode == NDARRAY_UINT16) return NDARRAY_UINT16;
        else return NDARRAY_INT16; // in2->array->typecode == NDARRAY_INT8, NDARRAY_INT16, 
    } else if(in1->array->typecode == NDARRAY_INT8) { // 3 cases
        return NDARRAY_INT16; // in2->array->typecode == NDARRAY_UINT8, NDARRAY_UINT16, NDARRAY_INT16
    } else if(in1->array->typecode == NDARRAY_UINT16) { // 3 cases
        if((in2->array->typecode == NDARRAY_UINT8) || (in2->array->typecode == NDARRAY_INT8)) return NDARRAY_UINT16;
        return NDARRAY_FLOAT; // in2->array->typecode == NDARRAY_INT16
    } else if(in1->array->typecode == NDARRAY_INT16) { // 3 cases
        if((in2->array->typecode == NDARRAY_UINT8) || (in2->array->typecode == NDARRAY_INT8)) return NDARRAY_INT16;
        return NDARRAY_FLOAT; // in2->array->typecode == NDARRAY_UINT16
    }
}

size_t *broadcasting(ndarray_obj_t *in1, ndarray_obj_t *in2) {
    // creates a new dense array with the dimensions
    // and shapes dictated by the broadcasting rules
    uint8_t ndim = in2->ndim;
    if(in1->ndim > in2->ndim) { 
        // overwrite ndim, if the first array is larger than the second
        ndim = in1->ndim;
    }
    size_t *shape = m_new(size_t, ndim);
    size_t *shape1 = m_new(size_t, ndim);
    size_t *shape2 = m_new(size_t, ndim);
    for(uint8_t i=0; i < ndim; i++) {
        // initialise the shapes with straight ones
        shape1[i] = shape2[i] = 1;
        shape[i] = 0;
    }
    // overwrite the first in1->ndim (in2->ndim) shape values from the right
    for(uint8_t i=ndim; i > ndim-in1->ndim; i--) {
        shape1[i-1] = in1->shape[i-1];
    }
    for(uint8_t i=ndim; i > ndim-in2->ndim; i--) {
        shape2[i-1] = in2->shape[i-1];
    }
    // check, whether the new shapes conform to the broadcasting rules
    for(uint8_t i=0; i < ndim; i++) {
        if((shape1[i] == shape2[i]) || (shape1[i] == 1) || (shape2[i] == 1)) {
            shape[i] = shape1[i];
            if(shape1[i] < shape2[i]) {
                shape[i] = shape2[i];
            }
        } else {
            m_del(size_t, shape1, ndim);
            m_del(size_t, shape2, ndim);
            break;
        }
    }
    m_del(size_t, shape1, ndim);
    m_del(size_t, shape2, ndim);
    return shape;
}

ndarray_obj_t *ndarray_do_binary_op(ndarray_obj_t *lhs, ndarray_obj_t *rhs, uint8_t op) {
    uint8_t typecode = upcasting(lhs, rhs);
    size_t *shape = broadcasting(lhs, rhs);
    uint8_t ndim = lhs->ndim;
    if(rhs->ndim > lhs->ndim) ndim = rhs->ndim;
    // this is going to be the result array
    ndarray_obj_t *ndarray = ndarray_new_dense_ndarray(ndim, shape, typecode);

    size_t offset = ndarray->offset;
    size_t roffset = rhs->offset;
    size_t loffset = lhs->offset;
    size_t *coords = ndarray_new_coords(ndim); // coordinates in ndarray
    size_t *lcoords = ndarray_new_coords(ndim); // coordinates in the left-hand-side ndarray
    size_t *rcoords = ndarray_new_coords(ndim); // coordinates in the right-hand-side ndarray
    int32_t *lstride = m_new(int32_t, ndim);
    int32_t *rstride = m_new(int32_t, ndim);
    for(uint8_t i=0; i < lhs->ndim; i++) {
        lstride[ndim-1-i] = lhs->strides[lhs->ndim-1-i];
    }
    for(uint8_t i=lhs->ndim; i < ndim; i++) {
        lstride[ndim-1-i] = lhs->strides[0];        
    }
    for(uint8_t i=0; i < rhs->ndim; i++) {
        rstride[ndim-1-i] = rhs->strides[rhs->ndim-1-i];
    }
    for(uint8_t i=rhs->ndim; i < ndim; i++) {
        rstride[ndim-1-i] = rhs->strides[0];        
    }
    
    mp_float_t lvalue, rvalue, result = 0.0;
    int32_t ilvalue, irvalue, iresult = 0;
    uint8_t float_size = sizeof(mp_float_t);
    void *narray = ndarray->array->items;
    uint8_t *larray = (uint8_t *)lhs->array->items;
    uint8_t *rarray = (uint8_t *)rhs->array->items;
    for(size_t i=0; i < ndarray->len; i++) {
        // we could make this prettier by moving everything into a function,
        // but the function call might be too expensive, especially that it is in the loop...
        // left hand side
        if(lhs->array->typecode == NDARRAY_UINT8) {
            ilvalue = (int32_t)((uint8_t *)larray)[loffset];
        } else if(lhs->array->typecode == NDARRAY_INT8) {
            ilvalue = (int32_t)((int8_t *)larray)[loffset];
        } else if(lhs->array->typecode == NDARRAY_UINT16) {
            ilvalue = (int32_t)((uint16_t *)larray)[loffset*2];
        } else if(lhs->array->typecode == NDARRAY_INT16) {
            ilvalue = (int32_t)((int16_t *)larray)[loffset*2];
        } else {
            lvalue = (mp_float_t)((mp_float_t *)larray)[loffset*float_size];
        }
        // right hand side
        if(rhs->array->typecode == NDARRAY_UINT8) {
            irvalue = (int32_t)((uint8_t *)rarray)[roffset];
        } else if(rhs->array->typecode == NDARRAY_INT8) {
            irvalue = (int32_t)((int8_t *)rarray)[roffset];
        } else if(rhs->array->typecode == NDARRAY_UINT16) {
            irvalue = (int32_t)((uint16_t *)rarray)[roffset*2];
        } else if(lhs->array->typecode == NDARRAY_INT16) {
            irvalue = (int32_t)((int16_t *)rarray)[roffset*2];
        } else {
            rvalue = (mp_float_t)((mp_float_t *)rarray)[roffset*float_size];
        }
        // this is the place, where the actual operations take place
        if(op == MP_BINARY_OP_ADD) {
            result = lvalue + rvalue;
        } else if(op == MP_BINARY_OP_SUBTRACT) {
            result = lvalue - rvalue;
        } else if(op == MP_BINARY_OP_MULTIPLY) {
            result = lvalue * rvalue;
        }
        // cast the result to the proper output type
        if(typecode == NDARRAY_UINT8) {
            ((uint8_t *)narray)[offset] = (uint8_t)result;
        } else if(typecode == NDARRAY_INT8) {
            ((int8_t *)narray)[offset] = (int8_t)result;
        } else if(typecode == NDARRAY_UINT16) {
            ((uint16_t *)narray)[offset] = (uint16_t)result;
        } else if(typecode == NDARRAY_INT16) {
            ((int16_t *)narray)[offset] = (int16_t)result;
        } else {
            ((mp_float_t *)narray)[offset] = result;
        }
        
        
        offset += ndarray->strides[ndim-1];
        coords[ndim-1] += 1;
        for(uint8_t j=ndim-1; j > 0; j--) {
            if(coords[j] == ndarray->shape[j]) {
                offset -= ndarray->shape[j] * ndarray->strides[j];
                offset += ndarray->strides[j-1];
                coords[j] = 0;
                coords[j-1] += 1;
            } else { // coordinates can change only, if the last coordinate changes
                break;
            }        
        }   
    }
    m_del(size_t, coords, ndim);
    m_del(size_t, lcoords, ndim);
    m_del(size_t, rcoords, ndim);
    m_del(int32_t, lstride, ndim);
    m_del(int32_t, rstride, ndim);    
    return ndarray;
}

// Binary operations
mp_obj_t ndarray_binary_op(mp_binary_op_t op, mp_obj_t LHS, mp_obj_t RHS) {
    // First, handle the case, when the operand on the right hand side is a scalar
    ndarray_obj_t *lhs = MP_OBJ_TO_PTR(LHS);
    ndarray_obj_t *rhs;
    if(mp_obj_is_int(RHS) || mp_obj_is_float(RHS)) {
        size_t *shape = m_new(size_t, lhs->ndim*sizeof(size_t));
        for(uint8_t i=0; i < lhs->ndim; i++) {
            shape[i] = 1;
        }
        if(mp_obj_is_int(RHS)) {
            int32_t ivalue = mp_obj_get_int(RHS);
            if((ivalue > 0) && (ivalue < 256)) {
                rhs = ndarray_new_dense_ndarray(lhs->ndim, shape, NDARRAY_UINT8);
            } else if((ivalue > 255) && (ivalue < 65535)) {
                rhs = ndarray_new_dense_ndarray(lhs->ndim, shape, NDARRAY_UINT16);
            } else if((ivalue < 0) && (ivalue > -128)) {
                rhs = ndarray_new_dense_ndarray(lhs->ndim, shape, NDARRAY_INT8);
            } else if((ivalue < -127) && (ivalue > -32767)) {
                rhs = ndarray_new_dense_ndarray(lhs->ndim, shape, NDARRAY_INT16);
            } else { // the integer value clearly does not fit the ulab types, so move on to float
                rhs = ndarray_new_dense_ndarray(lhs->ndim, shape, NDARRAY_FLOAT);
            }
            mp_binary_set_val_array(rhs->array->typecode, rhs->array->items, 0, RHS);
        } else { // we have a float
            rhs = ndarray_new_dense_ndarray(lhs->ndim, shape, NDARRAY_FLOAT);
            mp_binary_set_val_array(rhs->array->typecode, rhs->array->items, 0, RHS);
        }
    } else {
        // the right hand side is an ndarray
        rhs = MP_OBJ_TO_PTR(RHS);
    }
    
    ndarray_obj_t *ndarray = NULL;
    size_t *new_shape = broadcasting(lhs, rhs);
    
    switch(op) {
        case MP_BINARY_OP_MAT_MULTIPLY:
            break;
        
        case MP_BINARY_OP_ADD:
            if(new_shape[0] == 0) {
                mp_raise_ValueError("operands could not be cast together");
            }
            ndarray = ndarray_do_binary_op(lhs, rhs, op);
            // The parameters of RUN_BINARY_LOOP are 
            // typecode of type_out, type_left, type_right, out_array, lhs_array, rhs_array, shape, ndim, operator
            //RUN_BINARY_LOOP(ndarray, NDARRAY_FLOAT, mp_float_t, mp_float_t, mp_float_t, lhs, rhs, new_shape, ndim, operator);
            /* if(lhs->array->typecode == NDARRAY_UINT8) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_LOOP(NDARRAY_UINT8, uint8_t, uint8_t, uint8_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, uint8_t, int8_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint8_t, uint16_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, uint8_t, int16_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                    RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint8_t, mp_float_t, lhs, rhs, new_shape, ndim, operator);
                }
            } else if(lhs->array->typecode == NDARRAY_INT8) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, uint8_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_LOOP(NDARRAY_INT8, int8_t, int8_t, int8_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, uint16_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, int16_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                    RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, int8_t, mp_float_t, lhs, rhs, new_shape, ndim, operator);
                }                
            } else if(lhs->array->typecode == NDARRAY_UINT16) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, uint8_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, int8_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, uint16_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint16_t, int16_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                    RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint8_t, mp_float_t, lhs, rhs, new_shape, ndim, operator);
                }
            } else if(lhs->array->typecode == NDARRAY_INT16) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, uint8_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, int8_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, int16_t, uint16_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, int16_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                    RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint16_t, mp_float_t, lhs, rhs, new_shape, ndim, operator);
                }
            } else if(lhs->array->typecode == NDARRAY_FLOAT) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, uint8_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, int8_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, uint16_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, int16_t, lhs, rhs, new_shape, ndim, operator);
                } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                    RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, mp_float_t, lhs, rhs, new_shape, ndim, operator);
                } 
            }  else { // this should never happen
                mp_raise_TypeError("wrong input type");
            } */
        break;
        default:
//            m_del(size_t, shape, lhs->ndim*sizeof(size_t));
            return mp_const_none;
        }
    
    
    return MP_OBJ_FROM_PTR(ndarray);
//    return mp_const_none;
}

mp_obj_t ndarray_unary_op(mp_unary_op_t op, mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ndarray_obj_t *ndarray;

    switch(op) {
        case MP_UNARY_OP_LEN:
            return mp_obj_new_int(self->shape[0]);
            break;

        case MP_UNARY_OP_INVERT:
            if(self->array->typecode == NDARRAY_FLOAT) {
                mp_raise_ValueError("operation is not supported for given type");
            }
            // we can invert the content byte by byte, there is no need to distinguish between different typecodes
            ndarray = ndarray_copy_view(self, self->array->typecode);
            uint8_t *array = (uint8_t *)ndarray->array->items;
            if(self->boolean == NDARRAY_BOOLEAN) {
                for(size_t i=0; i < self->len; i++) array[i] = 1 - array[i];
            }
            else {
                for(size_t i=0; i < self->len; i++) array[i] = ~array[i];
            }
            return MP_OBJ_FROM_PTR(ndarray);
            break;
        
        case MP_UNARY_OP_NEGATIVE:
            if(self->boolean == NDARRAY_BOOLEAN) {
                mp_raise_TypeError("boolean negative '-' is not supported, use the '~' operator instead");
            }
            ndarray = ndarray_copy_view(self, self->array->typecode);
            if(self->array->typecode == NDARRAY_UINT8) {
                uint8_t *array = (uint8_t *)ndarray->array->items;
                for(size_t i=0; i < self->len; i++) array[i] = -array[i];
            } else if(self->array->typecode == NDARRAY_INT8) {
                int8_t *array = (int8_t *)ndarray->array->items;
                for(size_t i=0; i < self->len; i++) array[i] = -array[i];
            } else if(self->array->typecode == NDARRAY_UINT16) {
                uint16_t *array = (uint16_t *)ndarray->array->items;
                for(size_t i=0; i < self->len; i++) array[i] = -array[i];
            } else if(self->array->typecode == NDARRAY_INT16) {
                int16_t *array = (int16_t *)ndarray->array->items;
                for(size_t i=0; i < self->len; i++) array[i] = -array[i];
            } else {
                mp_float_t *array = (mp_float_t *)ndarray->array->items;
                for(size_t i=0; i < self->len; i++) array[i] = -array[i];
            }
            return MP_OBJ_FROM_PTR(ndarray);
            break;

        case MP_UNARY_OP_POSITIVE:
            return MP_OBJ_FROM_PTR(ndarray_copy_view(self, self->array->typecode));

        case MP_UNARY_OP_ABS:
            if((self->array->typecode == NDARRAY_UINT8) || (self->array->typecode == NDARRAY_UINT16)) {
                return MP_OBJ_FROM_PTR(ndarray_copy_view(self, self->array->typecode));
            }
            ndarray = ndarray_copy_view(self, self->array->typecode);
            if(self->array->typecode == NDARRAY_INT8) {
                int8_t *array = (int8_t *)ndarray->array->items;
                for(size_t i=0; i < self->len; i++) {
                    if(array[i] < 0) array[i] = -array[i];
                }
            } else if(self->array->typecode == NDARRAY_INT16) {
                int16_t *array = (int16_t *)ndarray->array->items;
                for(size_t i=0; i < self->len; i++) {
                    if(array[i] < 0) array[i] = -array[i];
                }
            } else {
                mp_float_t *array = (mp_float_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++) {
                    if(array[i] < 0) array[i] = -array[i];
                }
            }
            return MP_OBJ_FROM_PTR(ndarray);
        break;

        default: return MP_OBJ_NULL; // operator not supported
    }
}
