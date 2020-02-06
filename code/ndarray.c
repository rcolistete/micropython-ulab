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

// TODO: this function is used only in fft.c, and could be replaced by a macro
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

void ndarray_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind) {
    (void)kind;
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    size_t offset = self->offset;
    
    mp_print_str(print, "array([");
        
    for(size_t i=0; i < self->len; i++) {
        if(!self->boolean) {
            mp_obj_print_helper(print, mp_binary_get_val_array(self->array->typecode, self->array->items, offset), PRINT_REPR);
        } else {
            if(((uint8_t *)self->array->items)[offset]) {
                mp_print_str(print, "True");
            } else {
                mp_print_str(print, "False");
            }
        }
        if(i < self->len-1) mp_print_str(print, ", ");
        offset += self->stride;
    }
    if(self->boolean) {
        mp_print_str(print, "], dtype=bool)");
    } else if(self->array->typecode == NDARRAY_UINT8) {
        mp_print_str(print, "], dtype=uint8)");
    } else if(self->array->typecode == NDARRAY_INT8) {
        mp_print_str(print, "], dtype=int8)");
    } else if(self->array->typecode == NDARRAY_UINT16) {
        mp_print_str(print, "], dtype=uint16)");
    } else if(self->array->typecode == NDARRAY_INT16) {
        mp_print_str(print, "], dtype=int16)");
    } else if(self->array->typecode == NDARRAY_FLOAT) {
        mp_print_str(print, "], dtype=float)");
    }
}

ndarray_obj_t *ndarray_new_ndarray(size_t len, uint8_t typecode) {
    // Creates the base ndarray, and initialises the values to straight 0s
    ndarray_obj_t *ndarray = m_new_obj(ndarray_obj_t);
    ndarray->base.type = &ulab_ndarray_type;
    ndarray->stride = 1;
    ndarray->offset = 0;
    if(typecode == NDARRAY_BOOL) {
        typecode = NDARRAY_UINT8;
        ndarray->boolean = NDARRAY_BOOLEAN;
    } else {
        ndarray->boolean = NDARRAY_NUMERIC;
    }
    ndarray->len = len;
    mp_obj_array_t *array = array_new(typecode, ndarray->len);
    memset(array->items, 0, array->len); 
    ndarray->array = array;
    return ndarray;
}

ndarray_obj_t *ndarray_new_view(mp_obj_array_t *array, size_t len, int32_t stride, size_t offset, uint8_t boolean) {
    ndarray_obj_t *ndarray = m_new_obj(ndarray_obj_t);
    ndarray->base.type = &ulab_ndarray_type;
    ndarray->boolean = boolean;
    ndarray->stride = stride;
    ndarray->len = len;
    ndarray->offset = offset;
    ndarray->array = array;
    return ndarray;
}

ndarray_obj_t *ndarray_copy_view(ndarray_obj_t *input, uint8_t typecode) {
    ndarray_obj_t *ndarray = ndarray_new_ndarray(input->len, input->array->typecode);
    size_t offset = input->offset;
    mp_obj_t item;
    for(size_t i=0; i < ndarray->len; i++) {
        item = mp_binary_get_val_array(input->array->typecode, input->array->items, offset);
        mp_binary_set_val_array(typecode, ndarray->array->items, i, item);
        offset += ndarray->stride;
    }
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

mp_obj_t ndarray_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    // TODO: implement copy keyword
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
    self = ndarray_new_ndarray(len, dtype);
    
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

mp_obj_t ndarray_subscr(mp_obj_t self_in, mp_obj_t index, mp_obj_t value) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    
    if (value == MP_OBJ_SENTINEL) { // return value(s)
        if(mp_obj_is_int(index)) {
            return mp_binary_get_val_array(self->array->typecode, self->array->items, mp_obj_get_int(index));
        } else if(MP_OBJ_IS_TYPE(index, &mp_type_slice)) {
            mp_bound_slice_t slice;
            mp_seq_get_fast_slice_indexes(self->len, index, &slice);
            size_t len, correction;
            if(slice.step > 0) correction = -1;
            len = (slice.stop - slice.start + (slice.step + correction)) / slice.step;
            if(len < 0) len = 0;
            ndarray_obj_t *ndarray = ndarray_new_view(self->array, len, slice.step, self->offset+self->stride*slice.start, self->boolean);
            return MP_OBJ_FROM_PTR(ndarray);
        } else {
            mp_raise_msg(&mp_type_IndexError, translate("wrong index type"));
        }
    } else { // assignment; the value must be an ndarray, or a scalar
        return mp_const_none;        
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
    if(self->cur < ndarray->len) {
        // read the current value
        self->cur++;
        return mp_binary_get_val_array(ndarray->array->typecode, ndarray->array->items, self->cur-1);
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

mp_obj_t ndarray_strides(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(self->stride);
}

mp_obj_t ndarray_itemsize(mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(mp_binary_get_size('@', self->array->typecode, NULL));
}

// Binary operations
mp_obj_t ndarray_binary_op(mp_binary_op_t op, mp_obj_t LHS, mp_obj_t RHS) {
    // First, handle the case, when the operand on the right hand side is a scalar
    ndarray_obj_t *lhs = MP_OBJ_TO_PTR(LHS);
    ndarray_obj_t *rhs;
    if(mp_obj_is_int(RHS) || mp_obj_is_float(RHS)) {
        if(mp_obj_is_int(RHS)) {
            int32_t ivalue = mp_obj_get_int(RHS);
            if((ivalue > 0) && (ivalue < 256)) {
                rhs = ndarray_new_ndarray(1, NDARRAY_UINT8);
            } else if((ivalue > 255) && (ivalue < 65535)) {
                rhs = ndarray_new_ndarray(1, NDARRAY_UINT16);
            } else if((ivalue < 0) && (ivalue > -128)) {
                rhs = ndarray_new_ndarray(1, NDARRAY_INT8);
            } else if((ivalue < -127) && (ivalue > -32767)) {
                rhs = ndarray_new_ndarray(1, NDARRAY_INT16);
            } else { // the integer value clearly does not fit the ulab types, so move on to float
                rhs = ndarray_new_ndarray(1, NDARRAY_FLOAT);
            }
            mp_binary_set_val_array(rhs->array->typecode, rhs->array->items, 0, RHS);
        } else { // we have a float
            rhs = ndarray_new_ndarray(1, NDARRAY_FLOAT);
            mp_binary_set_val_array(rhs->array->typecode, rhs->array->items, 0, RHS);
        }
    } else {
        // the right hand side is an ndarray
        rhs = MP_OBJ_TO_PTR(RHS);
    }

    size_t len = lhs->len;
    if(rhs->len > len) len = rhs->len;
    
    // do not increment the offsets, if the array lenght is 1
    int32_t lstride = 0, rstride = 0;
    if(lhs->len > 1) lstride = lhs->stride;
    if(rhs->len > 1) rstride = rhs->stride;
    ndarray_obj_t *ndarray = NULL;
    
    switch(op) {
        case MP_BINARY_OP_OR:
        case MP_BINARY_OP_XOR:
        case MP_BINARY_OP_AND:
            if((lhs->array->typecode == NDARRAY_FLOAT) || (rhs->array->typecode == NDARRAY_FLOAT)) {
                mp_raise_TypeError("bitwise operations are not supported for input type");
            }
            if(lhs->array->typecode == NDARRAY_UINT8) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_UINT8, uint8_t, uint8_t, uint8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT8, int8_t, uint8_t, int8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_UINT16, uint16_t, uint8_t, uint16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT16, int16_t, uint8_t, int16_t, lhs, rhs, len, lstride, rstride, op);
                }
            } else if(lhs->array->typecode == NDARRAY_INT8) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT8, int8_t, int8_t, uint8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT8, int8_t, int8_t, int8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT16, int16_t, int8_t, int16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT16, int16_t, int16_t, int16_t, lhs, rhs, len, lstride, rstride, op);
                }
            } if(lhs->array->typecode == NDARRAY_UINT16) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_UINT16, uint16_t, uint16_t, uint8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT16, int16_t, uint16_t, int8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_UINT16, uint16_t, uint16_t, uint16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT16, int16_t, uint16_t, int16_t, lhs, rhs, len, lstride, rstride, op);
                }
            } if(lhs->array->typecode == NDARRAY_INT16) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT16, int16_t, int16_t, uint8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT16, int16_t, int16_t, int8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT16, int16_t, int16_t, uint16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_BITWISE_1D(ndarray, NDARRAY_INT16, int16_t, int16_t, int16_t, lhs, rhs, len, lstride, rstride, op);
                }
            } else { // this should never happen
                mp_raise_TypeError("wrong input type");
            }
            break;
        
        // RUN_BINARY_LOOP_1D(ndarray, typecode, type_out, type_left, type_right, lhs, rhs, len, lstride, rstride, operator)
        case MP_BINARY_OP_ADD:
        case MP_BINARY_OP_SUBTRACT:
        case MP_BINARY_OP_MULTIPLY:
        case MP_BINARY_OP_TRUE_DIVIDE:
        case MP_BINARY_OP_MODULO:
        case MP_BINARY_OP_POWER:
             if(lhs->array->typecode == NDARRAY_UINT8) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_UINT8, uint8_t, uint8_t, uint8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_INT16, int16_t, uint8_t, int8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_UINT16, uint16_t, uint8_t, uint16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_INT16, int16_t, uint8_t, int16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_FLOAT, mp_float_t, uint8_t, mp_float_t, lhs, rhs, len, lstride, rstride, op);
                }
            } else if(lhs->array->typecode == NDARRAY_INT8) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_INT16, int16_t, int8_t, uint8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_INT8, int8_t, int8_t, int8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_INT16, int16_t, int8_t, uint16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_INT16, int16_t, int8_t, int16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_FLOAT, mp_float_t, int8_t, mp_float_t, lhs, rhs, len, lstride, rstride, op);
                }                
            } else if(lhs->array->typecode == NDARRAY_UINT16) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_UINT16, uint16_t, uint16_t, uint8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_UINT16, uint16_t, uint16_t, int8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_UINT16, uint16_t, uint16_t, uint16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_FLOAT, mp_float_t, uint16_t, int16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_FLOAT, mp_float_t, uint8_t, mp_float_t, lhs, rhs, len, lstride, rstride, op);
                }
            } else if(lhs->array->typecode == NDARRAY_INT16) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_INT16, int16_t, int16_t, uint8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_INT16, int16_t, int16_t, int8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_FLOAT, mp_float_t, int16_t, uint16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_INT16, int16_t, int16_t, int16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_FLOAT, mp_float_t, uint16_t, mp_float_t, lhs, rhs, len, lstride, rstride, op);
                }
            } else if(lhs->array->typecode == NDARRAY_FLOAT) {
                if(rhs->array->typecode == NDARRAY_UINT8) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_FLOAT, mp_float_t, mp_float_t, uint8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT8) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_FLOAT, mp_float_t, mp_float_t, int8_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_UINT16) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_FLOAT, mp_float_t, mp_float_t, uint16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_INT16) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_FLOAT, mp_float_t, mp_float_t, int16_t, lhs, rhs, len, lstride, rstride, op);
                } else if(rhs->array->typecode == NDARRAY_FLOAT) {
                    RUN_BINARY_LOOP_1D(ndarray, NDARRAY_FLOAT, mp_float_t, mp_float_t, mp_float_t, lhs, rhs, len, lstride, rstride, op);
                } 
            }  else { // this should never happen
                mp_raise_TypeError("wrong input type");
            }
        break;
        default:
            return mp_const_none;
        }
    return MP_OBJ_FROM_PTR(ndarray);
}

mp_obj_t ndarray_unary_op(mp_unary_op_t op, mp_obj_t self_in) {
    ndarray_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ndarray_obj_t *ndarray;

    switch(op) {
        case MP_UNARY_OP_LEN:
            return mp_obj_new_int(self->len);
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
            } else {
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
