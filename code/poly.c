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
#include <string.h>
#include "py/obj.h"
#include "py/runtime.h"
#include "py/objarray.h"
#include "ndarray.h"
#include "linalg.h"
#include "poly.h"

#if ULAB_POLY_POLYVAL || ULAB_POLY_POLYFIT
void fill_array_iterable(mp_float_t *array, mp_obj_t oin) {
    mp_obj_iter_buf_t buf;
    mp_obj_t item, iterable = mp_getiter(oin, &buf);
    while((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
        *array++ = mp_obj_get_float(item);
    }
}
#endif

#if ULAB_POLY_POLYVAL
mp_obj_t poly_polyval(mp_obj_t o_p, mp_obj_t o_x) {
    // we always return floats: polynomials are going to be of type float, except, 
    // when both the coefficients and the independent variable are integers; 
    uint8_t plen = mp_obj_get_int(mp_obj_len_maybe(o_p));
    mp_float_t *p = m_new(mp_float_t, plen);
    fill_array_iterable(p, o_p);
    ndarray_obj_t *ndarray;
    mp_float_t *array;
    if(MP_OBJ_IS_TYPE(o_x, &ulab_ndarray_type)) {
        ndarray_obj_t *input = MP_OBJ_TO_PTR(o_x);
        ndarray = ndarray_copy_view(input, NDARRAY_FLOAT);
        array = (mp_float_t *)ndarray->array->items;
    } else { // at this point, we should have a 1-D iterable
        size_t len = mp_obj_get_int(mp_obj_len_maybe(o_x));
        ndarray = ndarray_new_ndarray(len, NDARRAY_FLOAT);
        array = (mp_float_t *)ndarray->array->items;
        fill_array_iterable(array, o_x);
    }
    mp_float_t x, y;
    for(size_t i=0; i < ndarray->len; i++) {
        x = array[i];
        y = p[0];
        for(uint8_t j=0; j < plen-1; j++) {
            y *= x;
            y += p[j+1];
        }
        array[i] = y;
    }
    m_del(mp_float_t, p, plen);
    return MP_OBJ_FROM_PTR(ndarray);
}

MP_DEFINE_CONST_FUN_OBJ_2(poly_polyval_obj, poly_polyval);
#endif

#if ULAB_POLY_POLYFIT
bool poly_invert_matrix(mp_float_t *data, size_t N) {
    // returns true, of the inversion was successful, 
    // false, if the matrix is singular
    
    // initially, this is the unit matrix: the contents of this matrix is what 
    // will be returned after all the transformations
    mp_float_t *unit = m_new(mp_float_t, N*N);

    mp_float_t elem = 1.0;
    // initialise the unit matrix
    memset(unit, 0, sizeof(mp_float_t)*N*N);
    for(size_t m=0; m < N; m++) {
        memcpy(&unit[m*(N+1)], &elem, sizeof(mp_float_t));
    }
    for(size_t m=0; m < N; m++){
        // this could be faster with ((c < epsilon) && (c > -epsilon))
        if(MICROPY_FLOAT_C_FUN(fabs)(data[m*(N+1)]) < epsilon) {
            m_del(mp_float_t, unit, N*N);
            return false;
        }
        for(size_t n=0; n < N; n++){
            if(m != n){
                elem = data[N*n+m] / data[m*(N+1)];
                for(size_t k=0; k < N; k++){
                    data[N*n+k] -= elem * data[N*m+k];
                    unit[N*n+k] -= elem * unit[N*m+k];
                }
            }
        }
    }
    for(size_t m=0; m < N; m++){ 
        elem = data[m*(N+1)];
        for(size_t n=0; n < N; n++){
            data[N*m+n] /= elem;
            unit[N*m+n] /= elem;
        }
    }
    memcpy(data, unit, sizeof(mp_float_t)*N*N);
    m_del(mp_float_t, unit, N*N);
    return true;
}

mp_obj_t poly_polyfit(size_t n_args, const mp_obj_t *args) {
    if(n_args != 3) {
        mp_raise_ValueError(translate("number of arguments must be 3"));
    }
    if(!MP_OBJ_IS_TYPE(args[0], &ulab_ndarray_type) && !MP_OBJ_IS_TYPE(args[0], &mp_type_tuple) &&
      !MP_OBJ_IS_TYPE(args[0], &mp_type_list) && !MP_OBJ_IS_TYPE(args[0], &mp_type_range) &&
      !MP_OBJ_IS_TYPE(args[1], &ulab_ndarray_type) && !MP_OBJ_IS_TYPE(args[1], &mp_type_tuple) &&
      !MP_OBJ_IS_TYPE(args[1], &mp_type_list) && !MP_OBJ_IS_TYPE(args[1], &mp_type_range)) {
        mp_raise_ValueError(translate("input data must be 1D iterable"));
    }
    uint16_t lenx, leny;
    uint8_t deg;
    mp_float_t *x, *XT, *y, *prod;

    lenx = (uint16_t)mp_obj_get_int(mp_obj_len_maybe(args[0]));
    leny = (uint16_t)mp_obj_get_int(mp_obj_len_maybe(args[1]));
    if(lenx != leny) {
        mp_raise_ValueError(translate("input vectors must be of equal length"));
    }
    deg = (uint8_t)mp_obj_get_int(args[2]);
    if(leny < deg) {
        mp_raise_ValueError(translate("more degrees of freedom than data points"));
    }
    x = m_new(mp_float_t, lenx);
    fill_array_iterable(x, args[0]);
    y = m_new(mp_float_t, leny);
    fill_array_iterable(y, args[1]);
    
    // one could probably express X as a function of XT, 
    // and thereby save RAM, because X is used only in the product
    XT = m_new(mp_float_t, (deg+1)*leny); // XT is a matrix of shape (deg+1, len) (rows, columns)
    for(uint8_t i=0; i < leny; i++) { // column index
        XT[i+0*lenx] = 1.0; // top row
        for(uint8_t j=1; j < deg+1; j++) { // row index
            XT[i+j*leny] = XT[i+(j-1)*leny]*x[i];
        }
    }
    
    prod = m_new(mp_float_t, (deg+1)*(deg+1)); // the product matrix is of shape (deg+1, deg+1)
    mp_float_t sum;
    for(uint16_t i=0; i < deg+1; i++) { // column index
        for(uint16_t j=0; j < deg+1; j++) { // row index
            sum = 0.0;
            for(size_t k=0; k < lenx; k++) {
                // (j, k) * (k, i) 
                // Note that the second matrix is simply the transpose of the first: 
                // X(k, i) = XT(i, k) = XT[k*lenx+i]
                sum += XT[j*lenx+k]*XT[i*lenx+k]; // X[k*(deg+1)+i];
            }
            prod[j*(deg+1)+i] = sum;
        }
    }
    if(!poly_invert_matrix(prod, deg+1)) {
        // Although X was a Vandermonde matrix, whose inverse is guaranteed to exist, 
        // we bail out here, if prod couldn't be inverted: if the values in x are not all 
        // distinct, prod is singular
        m_del(mp_float_t, XT, (deg+1)*lenx);
        m_del(mp_float_t, x, lenx);
        m_del(mp_float_t, y, lenx);
        m_del(mp_float_t, prod, (deg+1)*(deg+1));
        mp_raise_ValueError(translate("could not invert Vandermonde matrix"));
    } 
    // at this point, we have the inverse of X^T * X
    // y is a column vector; x is free now, we can use it for storing intermediate values
    for(uint16_t i=0; i < deg+1; i++) { // row index
        sum = 0.0;
        for(uint16_t j=0; j < lenx; j++) { // column index
            sum += XT[i*lenx+j]*y[j];
        }
        x[i] = sum;
    }
    // XT is no longer needed
    m_del(mp_float_t, XT, (deg+1)*leny);
    
    ndarray_obj_t *beta = ndarray_new_ndarray(deg+1, NDARRAY_FLOAT);
    mp_float_t *betav = (mp_float_t *)beta->array->items;
    // x[0..(deg+1)] contains now the product X^T * y; we can get rid of y
    m_del(float, y, leny);
    
    // now, we calculate beta, i.e., we apply prod = (X^T * X)^(-1) on x = X^T * y; x is a column vector now
    for(uint16_t i=0; i < deg+1; i++) {
        sum = 0.0;
        for(uint16_t j=0; j < deg+1; j++) {
            sum += prod[i*(deg+1)+j]*x[j];
        }
        betav[i] = sum;
    }
    m_del(mp_float_t, x, lenx);
    m_del(mp_float_t, prod, (deg+1)*(deg+1));
    for(uint8_t i=0; i < (deg+1)/2; i++) {
        // We have to reverse the array, for the leading coefficient comes first. 
        SWAP(mp_float_t, betav[i], betav[deg-i]);
    }
    return MP_OBJ_FROM_PTR(beta);
}

MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(poly_polyfit_obj, 2, 3, poly_polyfit);
#endif
