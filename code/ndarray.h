/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019-2020 Zoltán Vörös
*/
    
#ifndef _NDARRAY_
#define _NDARRAY_

#include "py/objarray.h"
#include "py/binary.h"
#include "py/objstr.h"
#include "py/objlist.h"

#define SWAP(t, a, b) { t tmp = a; a = b; b = tmp; }

#define NDARRAY_NUMERIC   0
#define NDARRAY_BOOLEAN   1

#if MICROPY_FLOAT_IMPL == MICROPY_FLOAT_IMPL_FLOAT
#define FLOAT_TYPECODE 'f'
#elif MICROPY_FLOAT_IMPL == MICROPY_FLOAT_IMPL_DOUBLE
#define FLOAT_TYPECODE 'd'
#endif

#if !CIRCUITPY
#define translate(x) x
#endif

extern const mp_obj_type_t ulab_ndarray_type;

enum NDARRAY_TYPE {
    NDARRAY_BOOL = '?', // this must never be assigned to the typecode!
    NDARRAY_UINT8 = 'B',
    NDARRAY_INT8 = 'b',
    NDARRAY_UINT16 = 'H', 
    NDARRAY_INT16 = 'h',
    NDARRAY_FLOAT = FLOAT_TYPECODE,
};

typedef struct _ndarray_obj_t {
    mp_obj_base_t base;
    uint8_t boolean;
    size_t len;
    int32_t stride;
    size_t offset;
    mp_obj_array_t *array;
} ndarray_obj_t;

mp_float_t ndarray_get_float_value(void *, uint8_t , size_t );

mp_obj_t mp_obj_new_ndarray_iterator(mp_obj_t , size_t , mp_obj_iter_buf_t *);
void ndarray_print(const mp_print_t *, mp_obj_t , mp_print_kind_t );
ndarray_obj_t *ndarray_new_ndarray(size_t , uint8_t );
ndarray_obj_t *ndarray_copy_view(ndarray_obj_t *, uint8_t );

mp_obj_t ndarray_copy(mp_obj_t );
mp_obj_t ndarray_make_new(const mp_obj_type_t *, size_t , size_t , const mp_obj_t *);
mp_obj_t ndarray_subscr(mp_obj_t , mp_obj_t , mp_obj_t );
mp_obj_t ndarray_getiter(mp_obj_t , mp_obj_iter_buf_t *);
mp_obj_t ndarray_binary_op(mp_binary_op_t , mp_obj_t , mp_obj_t );
mp_obj_t ndarray_unary_op(mp_unary_op_t , mp_obj_t );

mp_obj_t ndarray_strides(mp_obj_t );
mp_obj_t ndarray_itemsize(mp_obj_t );

#define RUN_BINARY_LOOP_1D(ndarray, typecode, type_out, type_left, type_right, lhs, rhs, len, lstride, rstride, operator) do {\
    type_left *left = (type_left *)(lhs)->array->items;\
    type_right *right = (type_right *)(rhs)->array->items;\
    (ndarray) = ndarray_new_ndarray((len), (typecode));\
    type_out *out = (type_out *)ndarray->array->items;\
    if((operator) == MP_BINARY_OP_ADD) {\
        for(size_t i=0; i < (len); i++, out++, left += (lstride), right += (rstride)) {\
            *out = (*left) + (*right);\
        }\
    } else if((operator) == MP_BINARY_OP_SUBTRACT) {\
        for(size_t i=0; i < (len); i++, left += (lstride), right += (rstride)) {\
            *out++ = (*left) - (*right);\
        }\
    } else if((operator) == MP_BINARY_OP_MULTIPLY) {\
        for(size_t i=0; i < (len); i++, left += (lstride), right += (rstride)) {\
            *out++ = (*left) * (*right);\
        }\
    } else if((operator) == MP_BINARY_OP_TRUE_DIVIDE) {\
        for(size_t i=0; i < (len); i++, left += (lstride), right += (rstride)) {\
            *out++ = (*left) / (*right);\
        }\
    } else if((operator) == MP_BINARY_OP_MODULO) {\
        for(size_t i=0; i < (len); i++, left += (lstride), right += (rstride)) {\
            *out++ = MICROPY_FLOAT_C_FUN(fmod)((*left), (*right));\
        }\
    } else if((operator) == MP_BINARY_OP_POWER) {\
        for(size_t i=0; i < (len); i++, left += (lstride), right += (rstride)) {\
            *out++ = MICROPY_FLOAT_C_FUN(pow)((*left), (*right));\
        }\
    }\
} while(0)

#define RUN_BINARY_BITWISE_1D(ndarray, typecode, type_out, type_left, type_right, lhs, rhs, len, lstride, rstride, operator) do {\
    type_left *left = (type_left *)(lhs)->array->items;\
    type_right *right = (type_right *)(rhs)->array->items;\
    (ndarray) = ndarray_new_ndarray((len), (typecode));\
    type_out *out = (type_out *)ndarray->array->items;\
    if((operator) == MP_BINARY_OP_OR) {\
        for(size_t i=0; i < (len); i++, out++, left += (lstride), right += (rstride)) {\
            *out = (*left) | (*right);\
        }\
    } else if((operator) == MP_BINARY_OP_XOR) {\
        for(size_t i=0; i < (len); i++, out++, left += (lstride), right += (rstride)) {\
            *out = (*left) ^ (*right);\
        }\
    } else if((operator) == MP_BINARY_OP_AND) {\
        for(size_t i=0; i < (len); i++, out++, left += (lstride), right += (rstride)) {\
            *out = (*left) & (*right);\
        }\
    }\
} while(0)

#endif
