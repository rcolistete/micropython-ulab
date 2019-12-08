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

// TODO: I am pretty sure there is a more elegant way of calculating the slice length...
size_t slice_length(mp_bound_slice_t slice) {
    if(slice.step == 1) {
        if(slice.start < slice.stop) return slice.stop - slice.start;
        else return 0;
    } else if(slice.step == -1) {
        if(slice.start > slice.stop) return slice.start - slice.stop;
        else return 0;
    } else {
        if(slice.step < 0) {
            if(slice.start < slice.stop) {
                return 0;
            }
            else {
                slice.step = -slice.step;
                return (slice.start - slice.stop + 1) / slice.step;
            }
        } else {
            if(slice.start < slice.stop) {
                return (slice.stop - slice.start + 1) / slice.step;
            } else {
                return 0;
            }
        }
    }
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
    
mp_obj_t ndarray_assign_view_from_tuple(ndarray_obj_t *ndarray, mp_obj_tuple_t *slices, mp_obj_t value) {
    // TODO: extend this to ndarrays
    
    ndarray_obj_t *result = ndarray_new_view_from_tuple(ndarray, slices);
    ndarray_obj_t *nvalue;
    if(MP_OBJ_IS_TYPE(value, &ulab_ndarray_type)) {
        nvalue = MP_OBJ_TO_PTR(value);
        if(nvalue->ndim > result->ndim) {
            mp_raise_ValueError("could not broadcast input array into output array");
        } for(uint8_t i=0; i < nvalue->ndim; i++) {
            if((nvalue->shape[nvalue->ndim-1-i] != 1) && 
               (nvalue->shape[nvalue->ndim-1-i] != result->shape[result->ndim-1-i])) {
                mp_raise_ValueError("could not broadcast input array into output array");
            }
        }
    } else { // we have a scalar, so create an ndarray for it
        size_t *shape = m_new(size_t, result->ndim*sizeof(size_t));
        for(uint8_t i=0; i < result->ndim; i++) {
            shape[i] = 1;
        }
        nvalue = ndarray_new_dense_ndarray(result->ndim, shape, result->array->typecode);
        mp_binary_set_val_array(nvalue->array->typecode, nvalue->array->items, 0, value);
    }
    size_t roffset = result->offset;
    size_t noffset = nvalue->offset;
    size_t *rcoords = ndarray_new_coords(result->ndim);
    mp_obj_t item;
    for(size_t i=0; i < result->len; i++) {
        item = mp_binary_get_val_array(nvalue->array->typecode, nvalue->array->items, noffset);
        mp_binary_set_val_array(result->array->typecode, result->array->items, roffset, item);
        roffset += result->strides[result->ndim-1];
        rcoords[result->ndim-1] += 1;
        if(nvalue->shape[result->ndim-1] != 1) {
            noffset += nvalue->strides[result->ndim-1];
        }
        for(uint8_t j=result->ndim-1; j > 0; j--) {
            if(rcoords[j] == result->shape[j]) {
                if(nvalue->shape[j] != 1) {
                    noffset -= nvalue->shape[j] * nvalue->strides[j];
                    noffset += nvalue->strides[j-1];                    
                }
                roffset -= result->shape[j] * result->strides[j];
                roffset += result->strides[j-1];
                rcoords[j] = 0;
                rcoords[j-1] += 1;
            } else { // coordinates can change only, if the last coordinate changes
                break;
            }
        }
    }
    m_del(size_t, rcoords, result->ndim);
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

/*
ndarray_obj_t *broadcasting(ndarray_obj_t *in1, ndarray_obj_t *in2) {
    // 
    uint8_t ndim = (in1->ndim > in2->ndim) ? in1->ndim : in2->ndim;    
    uint8_t dtype;
    ndarray_obj_t *ndarray;
    size_t *shape = m_new(size_t, ndim);
    if(in1->ndim < in2->ndim) {
        for(uint8_t i=0; i < ndim; i++) {
            //shape[i] = 
        }
    }
    ndarray_obj_t *ndarray = ndarray_new_dense_ndarray(ndim, shape, dtype);
    return ndarray;
}
*/
// Binary operations
mp_obj_t ndarray_binary_op(mp_binary_op_t op, mp_obj_t lhs, mp_obj_t rhs) {
    // First, handle the case, when one of the operands is a scalar
    ndarray_obj_t *RHS;
    if(mp_obj_is_int(rhs)) {
        int32_t ivalue = mp_obj_get_int(rhs);
        if((ivalue > 0) && (ivalue < 256)) {
            CREATE_SINGLE_ITEM(RHS, uint8_t, NDARRAY_UINT8, ivalue);
        } else if((ivalue > 255) && (ivalue < 65535)) {
            CREATE_SINGLE_ITEM(RHS, uint16_t, NDARRAY_UINT16, ivalue);
        } else if((ivalue < 0) && (ivalue > -128)) {
            CREATE_SINGLE_ITEM(RHS, int8_t, NDARRAY_INT8, ivalue);
        } else if((ivalue < -127) && (ivalue > -32767)) {
            CREATE_SINGLE_ITEM(RHS, int16_t, NDARRAY_INT16, ivalue);
        } else { // the integer value clearly does not fit the ulab types, so move on to float
            CREATE_SINGLE_ITEM(RHS, mp_float_t, NDARRAY_FLOAT, ivalue);
        }
    } else if(mp_obj_is_float(rhs)) {
        mp_float_t fvalue = mp_obj_get_float(rhs);        
        CREATE_SINGLE_ITEM(RHS, mp_float_t, NDARRAY_FLOAT, fvalue);
    } else {
        RHS = MP_OBJ_TO_PTR(rhs);
    }
    
//    return MP_OBJ_FROM_PTR(ndarray);
    return mp_const_none;
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
