
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
        array[i] = (mp_float_t)mp_obj_get_float(x_item);
        i++;
    }
}

void ndarray_iterative_print(const mp_print_t *print, ndarray_obj_t *ndarray, size_t *shape, int32_t *strides) {
    size_t offset = ndarray->offset;
    // TODO: we should store the length of the array in the ndarray header
    uint8_t print_extra = ndarray->ndim;
    size_t *coords = m_new(size_t, ndarray->ndim);
    for(size_t i=0; i < ndarray->len; i++) {
        for(uint8_t j=0; j < print_extra; j++) {
            printf("[");
        }
        print_extra = 0;
        mp_obj_print_helper(print, mp_binary_get_val_array(ndarray->array->typecode, ndarray->array->items, offset), PRINT_REPR);
        offset += ndarray->strides[ndarray->ndim-1];
        coords[ndarray->ndim-1] += 1;
        if(coords[ndarray->ndim-1] != ndarray->shape[ndarray->ndim-1]) {
            mp_print_str(print, ", ");
        }
        for(uint8_t j=ndarray->ndim-1; j > 0; j--) {
            if(coords[j] == ndarray->shape[j]) {
                offset -= ndarray->shape[j]*ndarray->strides[j];
                offset += ndarray->strides[j-1];
                print_extra += 1;
                coords[j] = 0;
                coords[j-1] += 1;
                printf("]");
            }
        }
        if(print_extra && (i != ndarray->len-1)) {
            printf(",\n");
            if(print_extra > 1) {
                printf("\n");
            }
        }
    }
    printf("]");
    m_del(size_t, coords, ndarray->ndim);
}

void ndarray_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind) {
    (void)kind;
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    mp_print_str(print, "array(");
    ndarray_iterative_print(print, self, self->shape, self->strides);
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
        strides[i-1] = strides[i] * shape[i-1];
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

ndarray_obj_t *ndarray_new_linear_array(size_t len, uint8_t dtype) {
    size_t *shape = m_new(size_t, 1);
    int32_t *strides = m_new(int32_t, 1);
    shape[0] = len;
    strides[0] = 1;
    return ndarray_new_ndarray(1, shape, strides, dtype);
}
ndarray_obj_t *ndarray_new_linear_array(size_t len, uint8_t dtype) {
    size_t *shape = m_new(size_t, 1);
    int32_t *strides = m_new(int32_t, 1);
    shape[0] = len;
    strides[0] = 1;
    return ndarray_new_ndarray(1, shape, strides, dtype);
}

STATIC mp_obj_t ndarray_make_new_core(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args, mp_map_t *kw_args) {
    uint8_t dtype = ndarray_init_helper(n_args, args, kw_args);

    size_t len1, len2=0, i=0;
    mp_obj_t len_in = mp_obj_len_maybe(args[0]);
    if (len_in == MP_OBJ_NULL) {
        mp_raise_ValueError(translate("first argument must be an iterable"));
    } else {
        // len1 is either the number of rows (for matrices), or the number of elements (row vectors)
        len1 = MP_OBJ_SMALL_INT_VALUE(len_in);
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
/*
mp_obj_t ndarray_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    // TODO: implement dtype, and copy keywords
    mp_arg_check_num(n_args, n_kw, 1, 2, true);
    mp_map_t kw_args;
    mp_map_init_fixed_table(&kw_args, n_kw, args + n_args);
    uint8_t dtype = ndarray_init_helper(n_args, args, &kw_args);

    mp_obj_t len_in = mp_obj_len_maybe(args[0]);
    size_t len = MP_OBJ_SMALL_INT_VALUE(len_in);

    // work with a single dimension for now
    ndarray_obj_t *self = ndarray_new_linear_array(len, dtype);
    
    size_t i = 0;
    mp_obj_iter_buf_t iter_buf;
    mp_obj_t item, iterable = mp_getiter(args[0], &iter_buf);
    while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
        mp_binary_set_val_array(dtype, self->array->items, i++, item);
    }
    return MP_OBJ_FROM_PTR(self);
}
*/

mp_obj_t ndarray_subscr(mp_obj_t self_in, mp_obj_t index, mp_obj_t value) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    
    if (value == MP_OBJ_SENTINEL) { // return value(s)
        return ndarray_get_slice(self, index, NULL);    
    } else { // assignment to slices; the value must be an ndarray, or a scalar
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

mp_obj_t ndarray_size(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(self->array->len);
}

mp_obj_t ndarray_itemsize(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return MP_OBJ_NEW_SMALL_INT(mp_binary_get_size('@', self->array->typecode, NULL));
}


// Binary operations

mp_obj_t ndarray_binary_op(mp_binary_op_t op, mp_obj_t lhs, mp_obj_t rhs) {
//    if(op == MP_BINARY_OP_REVERSE_ADD) {
 //       return ndarray_binary_op(MP_BINARY_OP_ADD, rhs, lhs);
  //  }    
    // One of the operands is a scalar
    // TODO: conform to numpy with the upcasting
    // TODO: implement in-place operators
    mp_obj_t RHS = MP_OBJ_NULL;
    bool rhs_is_scalar = true;
    if(MP_OBJ_IS_INT(rhs)) {
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
        RHS = rhs;
        rhs_is_scalar = false;
    }
    //else 
    if(MP_OBJ_IS_TYPE(lhs, &ulab_ndarray_type) && MP_OBJ_IS_TYPE(RHS, &ulab_ndarray_type)) { 
        // next, the ndarray stuff
        ndarray_obj_t *ol = MP_OBJ_TO_PTR(lhs);
        ndarray_obj_t *or = MP_OBJ_TO_PTR(RHS);
        if(!rhs_is_scalar && ((ol->m != or->m) || (ol->n != or->n))) {
            mp_raise_ValueError(translate("operands could not be broadcast together"));
        }
        // At this point, the operands should have the same shape
        switch(op) {
            case MP_BINARY_OP_EQUAL:
                // Two arrays are equal, if their shape, typecode, and elements are equal
                if((ol->m != or->m) || (ol->n != or->n) || (ol->array->typecode != or->array->typecode)) {
                    return mp_const_false;
                } else {
                    size_t i = ol->bytes;
                    uint8_t *l = (uint8_t *)ol->array->items;
                    uint8_t *r = (uint8_t *)or->array->items;
                    while(i) { // At this point, we can simply compare the bytes, the type is irrelevant
                        if(*l++ != *r++) {
                            return mp_const_false;
                        }
                        i--;
                    }
                    return mp_const_true;
                }
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
                // uint8 + int8 => int16
                // uint8 + int16 => int16
                // uint8 + uint16 => uint16
                // int8 + int16 => int16
                // int8 + uint16 => uint16
                // uint16 + int16 => float
                // The parameters of RUN_BINARY_LOOP are 
                // typecode of result, type_out, type_left, type_right, lhs operand, rhs operand, operator
                if(ol->array->typecode == NDARRAY_UINT8) {
                    if(or->array->typecode == NDARRAY_UINT8) {
                        RUN_BINARY_LOOP(NDARRAY_UINT8, uint8_t, uint8_t, uint8_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_INT8) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, uint8_t, int8_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_UINT16) {
                        RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint8_t, uint16_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_INT16) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, uint8_t, int16_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_FLOAT) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint8_t, mp_float_t, ol, or, op);
                    }
                } else if(ol->array->typecode == NDARRAY_INT8) {
                    if(or->array->typecode == NDARRAY_UINT8) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, uint8_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_INT8) {
                        RUN_BINARY_LOOP(NDARRAY_INT8, int8_t, int8_t, int8_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_UINT16) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, uint16_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_INT16) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int8_t, int16_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_FLOAT) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, int8_t, mp_float_t, ol, or, op);
                    }                
                } else if(ol->array->typecode == NDARRAY_UINT16) {
                    if(or->array->typecode == NDARRAY_UINT8) {
                        RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, uint8_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_INT8) {
                        RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, int8_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_UINT16) {
                        RUN_BINARY_LOOP(NDARRAY_UINT16, uint16_t, uint16_t, uint16_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_INT16) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint16_t, int16_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_FLOAT) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint8_t, mp_float_t, ol, or, op);
                    }
                } else if(ol->array->typecode == NDARRAY_INT16) {
                    if(or->array->typecode == NDARRAY_UINT8) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, uint8_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_INT8) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, int8_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_UINT16) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, int16_t, uint16_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_INT16) {
                        RUN_BINARY_LOOP(NDARRAY_INT16, int16_t, int16_t, int16_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_FLOAT) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, uint16_t, mp_float_t, ol, or, op);
                    }
                } else if(ol->array->typecode == NDARRAY_FLOAT) {
                    if(or->array->typecode == NDARRAY_UINT8) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, uint8_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_INT8) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, int8_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_UINT16) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, uint16_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_INT16) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, int16_t, ol, or, op);
                    } else if(or->array->typecode == NDARRAY_FLOAT) {
                        RUN_BINARY_LOOP(NDARRAY_FLOAT, mp_float_t, mp_float_t, mp_float_t, ol, or, op);
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

mp_obj_t ndarray_unary_op(mp_unary_op_t op, mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ndarray_obj_t *ndarray = NULL;
    switch (op) {
        case MP_UNARY_OP_LEN: 
            if(self->m > 1) {
                return mp_obj_new_int(self->m);
            } else {
                return mp_obj_new_int(self->n);
            }
            break;
        
        case MP_UNARY_OP_INVERT:
            if(self->array->typecode == NDARRAY_FLOAT) {
                mp_raise_ValueError(translate("operation is not supported for given type"));
            }
            // we can invert the content byte by byte, there is no need to distinguish 
            // between different typecodes
            ndarray = MP_OBJ_TO_PTR(ndarray_copy(self_in));
            uint8_t *array = (uint8_t *)ndarray->array->items;
            for(size_t i=0; i < self->bytes; i++) array[i] = ~array[i];
            return MP_OBJ_FROM_PTR(ndarray);
            break;
        
        case MP_UNARY_OP_NEGATIVE:
            ndarray = MP_OBJ_TO_PTR(ndarray_copy(self_in));
            if(self->array->typecode == NDARRAY_UINT8) {
                uint8_t *array = (uint8_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++) array[i] = -array[i];
            } else if(self->array->typecode == NDARRAY_INT8) {
                int8_t *array = (int8_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++) array[i] = -array[i];
            } else if(self->array->typecode == NDARRAY_UINT16) {                
                uint16_t *array = (uint16_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++) array[i] = -array[i];
            } else if(self->array->typecode == NDARRAY_INT16) {
                int16_t *array = (int16_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++) array[i] = -array[i];
            } else {
                mp_float_t *array = (mp_float_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++) array[i] = -array[i];
            }
            return MP_OBJ_FROM_PTR(ndarray);
            break;

        case MP_UNARY_OP_POSITIVE:
            return ndarray_copy(self_in);

        case MP_UNARY_OP_ABS:
            if((self->array->typecode == NDARRAY_UINT8) || (self->array->typecode == NDARRAY_UINT16)) {
                return ndarray_copy(self_in);
            }
            ndarray = MP_OBJ_TO_PTR(ndarray_copy(self_in));
            if(self->array->typecode == NDARRAY_INT8) {
                int8_t *array = (int8_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++) {
                    if(array[i] < 0) array[i] = -array[i];
                }
            } else if(self->array->typecode == NDARRAY_INT16) {
                int16_t *array = (int16_t *)ndarray->array->items;
                for(size_t i=0; i < self->array->len; i++) {
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

mp_obj_t ndarray_transpose(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    // the size of a single item in the array
    uint8_t _sizeof = mp_binary_get_size('@', self->array->typecode, NULL);
    
    // NOTE: 
    // if the matrices are square, we can simply swap items, but 
    // generic matrices can't be transposed in place, so we have to 
    // declare a temporary variable
    
    // NOTE: 
    //  In the old matrix, the coordinate (m, n) is m*self->n + n
    //  We have to assign this to the coordinate (n, m) in the new 
    //  matrix, i.e., to n*self->m + m (since the new matrix has self->m columns)
    
    // one-dimensional arrays can be transposed by simply swapping the dimensions
    if((self->m != 1) && (self->n != 1)) {
        uint8_t *c = (uint8_t *)self->array->items;
        // self->bytes is the size of the bytearray, irrespective of the typecode
        uint8_t *tmp = m_new(uint8_t, self->bytes);
        for(size_t m=0; m < self->m; m++) {
            for(size_t n=0; n < self->n; n++) {
                memcpy(tmp+_sizeof*(n*self->m + m), c+_sizeof*(m*self->n + n), _sizeof);
            }
        }
        memcpy(self->array->items, tmp, self->bytes);
        m_del(uint8_t, tmp, self->bytes);
    } 
    SWAP(size_t, self->m, self->n);
    return mp_const_none;
}

MP_DEFINE_CONST_FUN_OBJ_1(ndarray_transpose_obj, ndarray_transpose);

mp_obj_t ndarray_reshape(mp_obj_t self_in, mp_obj_t shape) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if(!MP_OBJ_IS_TYPE(shape, &mp_type_tuple) || (MP_OBJ_SMALL_INT_VALUE(mp_obj_len_maybe(shape)) != 2)) {
        mp_raise_ValueError(translate("shape must be a 2-tuple"));
    }

    mp_obj_iter_buf_t iter_buf;
    mp_obj_t item, iterable = mp_getiter(shape, &iter_buf);
    uint16_t m, n;
    item = mp_iternext(iterable);
    m = mp_obj_get_int(item);
    item = mp_iternext(iterable);
    n = mp_obj_get_int(item);
    if(m*n != self->m*self->n) {
        // TODO: the proper error message would be "cannot reshape array of size %d into shape (%d, %d)"
        mp_raise_ValueError(translate("cannot reshape array (incompatible input/output shape)"));
    }
    self->m = m;
    self->n = n;
    return MP_OBJ_FROM_PTR(self);
}

MP_DEFINE_CONST_FUN_OBJ_2(ndarray_reshape_obj, ndarray_reshape);

mp_int_t ndarray_get_buffer(mp_obj_t self_in, mp_buffer_info_t *bufinfo, mp_uint_t flags) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    // buffer_p.get_buffer() returns zero for success, while mp_get_buffer returns true for success
    return !mp_get_buffer(self->array, bufinfo, flags);
}
