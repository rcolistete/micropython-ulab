/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 Zoltán Vörös
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

const mp_obj_type_t ulab_ndarray_type;

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
    uint8_t ndim;
    size_t *shape;
    int32_t *strides;
    size_t len;
    size_t offset;
    mp_obj_array_t *array;
} ndarray_obj_t;

// this is a helper structure, so that we can return shape AND strides from a function
typedef struct _ndarray_header_obj_t {
    size_t *shape;
    int32_t *strides;
    int8_t axis;
    size_t offset;
} ndarray_header_obj_t;

// various helper functions
size_t ndarray_index_from_flat(size_t , ndarray_obj_t *, int32_t *);
size_t ndarray_index_from_contracted(size_t  , ndarray_obj_t * , int32_t * , uint8_t  , uint8_t  );
mp_float_t ndarray_get_float_value(void *, uint8_t , size_t );

// calculates the index (in the original linear array) of an item, if the index in the flat array is given
// this is the macro equivalent of ndarray_index_from_flat()
// TODO: This fails, when the last stride is not 1!!!
#define NDARRAY_INDEX_FROM_FLAT(ndarray, shape_strides, index, _tindex, _nindex) do {\
    size_t Q;\
    (_tindex) = (index);\
    (_nindex) = (ndarray)->offset;\
    for(size_t _x=0; _x < (ndarray)->ndim; _x++) {\
        Q = (_tindex) / (shape_strides)[_x];\
        (_tindex) -= Q * (shape_strides)[_x];\
        (_nindex) += Q * (ndarray)->strides[_x];\
    }\
} while(0)

#define NDARRAY_INDEX_FROM_FLAT2(ndarray, stride_array, shape_strides, index, _tindex, _nindex) do {\
    size_t Q;\
    (_tindex) = (index);\
    (_nindex) = (ndarray)->offset;\
    for(size_t _x=0; _x < (ndarray)->ndim; _x++) {\
        Q = (_tindex) / (shape_strides)[_x];\
        (_tindex) -= Q * (shape_strides)[_x];\
        (_nindex) += Q * (stride_array)[_x];\
    }\
} while(0)
    
#define CREATE_SINGLE_ITEM(ndarray, type, typecode, value) do {\
    (ndarray) = ndarray_new_linear_array(1, (typecode));\
    type *tmparr = (type *)(ndarray)->array->items;\
    tmparr[0] = (type)(value);\
} while(0)

mp_obj_t mp_obj_new_ndarray_iterator(mp_obj_t , size_t , mp_obj_iter_buf_t *);
void ndarray_print(const mp_print_t *, mp_obj_t , mp_print_kind_t );
ndarray_obj_t *ndarray_new_ndarray(uint8_t , size_t *, int32_t *, uint8_t );
ndarray_obj_t *ndarray_new_dense_ndarray(uint8_t , size_t *, uint8_t );
ndarray_obj_t *ndarray_new_linear_array(size_t , uint8_t );
ndarray_obj_t *ndarray_copy_view(ndarray_obj_t *, uint8_t );

mp_obj_t ndarray_copy(mp_obj_t );
mp_obj_t ndarray_make_new(const mp_obj_type_t *, size_t , size_t , const mp_obj_t *);
mp_obj_t ndarray_subscr(mp_obj_t , mp_obj_t , mp_obj_t );
mp_obj_t ndarray_getiter(mp_obj_t , mp_obj_iter_buf_t *);
mp_obj_t ndarray_binary_op(mp_binary_op_t , mp_obj_t , mp_obj_t );
mp_obj_t ndarray_unary_op(mp_unary_op_t , mp_obj_t );

mp_obj_t ndarray_shape(mp_obj_t );
mp_obj_t ndarray_reshape(mp_obj_t , mp_obj_t );
mp_obj_t ndarray_transpose(mp_obj_t );
mp_obj_t ndarray_flatten(size_t , const mp_obj_t *, mp_map_t *);
mp_obj_t ndarray_itemsize(mp_obj_t );
mp_obj_t ndarray_strides(mp_obj_t );
mp_obj_t ndarray_info(mp_obj_t );

#endif
