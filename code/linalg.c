/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 Zoltán Vörös
*/
    
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "py/obj.h"
#include "py/runtime.h"
#include "py/misc.h"
#include "linalg.h"

bool linalg_invert_matrix(mp_float_t *data, size_t N) {
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

mp_obj_t linalg_inv(mp_obj_t o_in) {
    if(!MP_OBJ_IS_TYPE(o_in, &ulab_ndarray_type)) {
        mp_raise_TypeError("only ndarray objects can be inverted");
    }
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(o_in);
    if(ndarray->ndim != 2) {
        mp_raise_ValueError("only two-dimensional tensors can be inverted");
    }
    if(ndarray->shape[0] != ndarray->shape[1]) {
        mp_raise_ValueError("only square matrices can be inverted");
    }
    size_t *shape = m_new(size_t, 2);
    shape[0] = shape[1] = ndarray->shape[0];
    ndarray_obj_t *inverted = ndarray_new_dense_ndarray(2, shape, NDARRAY_FLOAT);
    mp_float_t *data = (mp_float_t *)inverted->array->items;
    mp_obj_t elem;
    for(size_t m=0; m < ndarray->shape[0]; m++) { // rows first
        for(size_t n=0; n < ndarray->shape[1]; n++) { // columns next
            // this could, perhaps, be done in single line... 
            // On the other hand, we probably spend little time here
            elem = mp_binary_get_val_array(ndarray->array->typecode, ndarray->array->items, m*ndarray->shape[1]+n);
            data[m*ndarray->shape[1]+n] = (mp_float_t)mp_obj_get_float(elem);
        }
    }
    
    if(!linalg_invert_matrix(data, ndarray->shape[0])) {
        // TODO: I am not sure this is needed here. Otherwise, how should we free up the unused RAM of inverted?
        m_del(mp_float_t, inverted->array->items, ndarray->shape[0]*ndarray->shape[1]);
        mp_raise_ValueError("input matrix is singular");
    }
    return MP_OBJ_FROM_PTR(inverted);
}

mp_obj_t linalg_dot(mp_obj_t _m1, mp_obj_t _m2) {
    return mp_const_none;
    /*
    // TODO: should the results be upcast?
    ndarray_obj_t *m1 = MP_OBJ_TO_PTR(_m1);
    ndarray_obj_t *m2 = MP_OBJ_TO_PTR(_m2);    
    if(m1->n != m2->m) {
        mp_raise_ValueError("matrix dimensions do not match");
    }
    // TODO: numpy uses upcasting here
    ndarray_obj_t *out = create_new_ndarray(m1->m, m2->n, NDARRAY_FLOAT);
    mp_float_t *outdata = (mp_float_t *)out->array->items;
    mp_float_t sum, v1, v2;
    for(size_t i=0; i < m1->m; i++) { // rows of m1
        for(size_t j=0; j < m2->n; j++) { // columns of m2
            sum = 0.0;
            for(size_t k=0; k < m2->m; k++) {
                // (i, k) * (k, j)
                v1 = ndarray_get_float_value(m1->array->items, m1->array->typecode, i*m1->n+k);
                v2 = ndarray_get_float_value(m2->array->items, m2->array->typecode, k*m2->n+j);
                sum += v1 * v2;
            }
            outdata[i*m1->m+j] = sum;
        }
    }
    return MP_OBJ_FROM_PTR(out); */
} 

mp_obj_t linalg_zeros_ones(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args, uint8_t kind) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL} } ,
        { MP_QSTR_dtype, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = NDARRAY_FLOAT} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    
    uint8_t dtype = args[1].u_int;
    if(!mp_obj_is_int(args[0].u_obj) && !mp_obj_is_type(args[0].u_obj, &mp_type_tuple)) {
        mp_raise_TypeError("input argument must be an integer or a tuple");
    }
    ndarray_obj_t *ndarray = NULL;
    if(mp_obj_is_int(args[0].u_obj)) {
        size_t n = mp_obj_get_int(args[0].u_obj);
        size_t *shape = m_new(size_t, 1);
        shape[0] = n;
        ndarray = ndarray_new_dense_ndarray(1, shape, dtype);
    } else if(mp_obj_is_type(args[0].u_obj, &mp_type_tuple)) {
        mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(args[0].u_obj);
        size_t *shape = m_new(size_t, tuple->len);
        for(uint8_t i=0; i < tuple->len; i++) {
            shape[i] = mp_obj_get_int(tuple->items[i]);
        }
        ndarray = ndarray_new_dense_ndarray(tuple->len, shape, dtype);
    }
    if(kind == 1) {
        mp_obj_t one = mp_obj_new_int(1);
        for(size_t i=0; i < ndarray->array->len; i++) {
            mp_binary_set_val_array(dtype, ndarray->array->items, i, one);
        }
    }
    return MP_OBJ_FROM_PTR(ndarray);
}

mp_obj_t linalg_zeros(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return linalg_zeros_ones(n_args, pos_args, kw_args, 0);
}

mp_obj_t linalg_ones(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return linalg_zeros_ones(n_args, pos_args, kw_args, 1);
}

mp_obj_t linalg_eye(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    // TODO: this is a bit more problematic in higher dimensions
    return mp_const_none;
    /*
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_M, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_PTR(&mp_const_none_obj) } },
        { MP_QSTR_k, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = 0} },        
        { MP_QSTR_dtype, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = NDARRAY_FLOAT} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    size_t n = args[0].u_int, m;
    int16_t k = args[2].u_int;
    uint8_t dtype = args[3].u_int;
    if(args[1].u_rom_obj == mp_const_none) {
        m = n;
    } else {
        m = mp_obj_get_int(args[1].u_rom_obj);
    }
    
    ndarray_obj_t *ndarray = create_new_ndarray(m, n, dtype);
    mp_obj_t one = mp_obj_new_int(1);
    size_t i = 0;
    if((k >= 0) && (k < n)) {
        while(k < n) {
            mp_binary_set_val_array(dtype, ndarray->array->items, i*n+k, one);
            k++;
            i++;
        }
    } else if((k < 0) && (-k < m)) {
        k = -k;
        i = 0;
        while(k < m) {
            mp_binary_set_val_array(dtype, ndarray->array->items, k*n+i, one);
            k++;
            i++;
        }
    }
    return MP_OBJ_FROM_PTR(ndarray); */
}

mp_obj_t linalg_det(mp_obj_t oin) {
    if(!mp_obj_is_type(oin, &ulab_ndarray_type)) {
        mp_raise_TypeError("function defined for ndarrays only");
    }
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(oin);
    if(ndarray->ndim != 2) {
        mp_raise_ValueError("only two-dimensional tensors can be inverted");
    }
    if(ndarray->shape[0] != ndarray->shape[1]) {
        mp_raise_ValueError("only square matrices can be inverted");
    }
    
    mp_float_t *tmp = m_new(mp_float_t, ndarray->shape[0]*ndarray->shape[1]);
    // TODO: this won't work for sliced arrays
    for(size_t i=0; i < ndarray->len; i++){
        tmp[i] = ndarray_get_float_value(ndarray->array->items, ndarray->array->typecode, i);
    }
    mp_float_t c;
    for(size_t m=0; m < ndarray->shape[0]-1; m++){
        if(MICROPY_FLOAT_C_FUN(fabs)(tmp[m*(ndarray->shape[1]+1)]) < epsilon) {
            m_del(mp_float_t, tmp, ndarray->shape[0]*ndarray->shape[1]);
            return mp_obj_new_float(0.0);
        }
        for(size_t n=0; n < ndarray->shape[1]; n++){
            if(m != n) {
                c = tmp[ndarray->shape[0]*n+m] / tmp[m*(ndarray->shape[1]+1)];
                for(size_t k=0; k < ndarray->shape[1]; k++){
                    tmp[ndarray->shape[1]*n+k] -= c * tmp[ndarray->shape[1]*m+k];
                }
            }
        }
    }
    mp_float_t det = 1.0;
                            
    for(size_t m=0; m < ndarray->shape[0]; m++){ 
        det *= tmp[m*(ndarray->shape[1]+1)];
    }
    m_del(mp_float_t, tmp, ndarray->shape[0]*ndarray->shape[1]);
    return mp_obj_new_float(det);
}

mp_obj_t linalg_eig(mp_obj_t oin) {
    if(!mp_obj_is_type(oin, &ulab_ndarray_type)) {
        mp_raise_TypeError("function defined for ndarrays only");
    }
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(oin);
    if(ndarray->ndim != 2) {
        mp_raise_ValueError("only two-dimensional tensors can be inverted");
    }
    if(ndarray->shape[0] != ndarray->shape[1]) {
        mp_raise_ValueError("only square matrices can be inverted");
    }
    mp_float_t *array = m_new(mp_float_t, ndarray->len);
    // TODO: this won't work for sliced arrays
    for(size_t i=0; i < ndarray->len; i++) {
        array[i] = ndarray_get_float_value(ndarray->array->items, ndarray->array->typecode, i);
    }
    // make sure the matrix is symmetric
    for(size_t m=0; m < ndarray->shape[0]; m++) {
        for(size_t n=m+1; n < ndarray->shape[1]; n++) {
            // compare entry (m, n) to (n, m)
            // TODO: this must probably be scaled!
            if(epsilon < MICROPY_FLOAT_C_FUN(fabs)(array[m*ndarray->shape[0] + n] - array[n*ndarray->shape[0] + m])) {
                mp_raise_ValueError("input matrix is asymmetric");
            }
        }
    }
    
    // if we got this far, then the matrix will be symmetric
    size_t *shape = m_new(size_t, 2);
    shape[0] = shape[1] = ndarray->shape[0];
    ndarray_obj_t *eigenvectors = ndarray_new_dense_ndarray(2, shape, NDARRAY_FLOAT);
    mp_float_t *eigvectors = (mp_float_t *)eigenvectors->array->items;
    // start out with the unit matrix
    for(size_t m=0; m < ndarray->shape[0]; m++) {
        eigvectors[m*(ndarray->shape[1]+1)] = 1.0;
    }
    mp_float_t largest, w, t, c, s, tau, aMk, aNk, vm, vn;
    size_t M, N;
    size_t iterations = JACOBI_MAX*ndarray->shape[0]*ndarray->shape[0];
    do {
        iterations--;
        // find the pivot here
        M = 0;
        N = 0;
        largest = 0.0;
        for(size_t m=0; m < ndarray->shape[0]-1; m++) { // -1: no need to inspect last row
            for(size_t n=m+1; n < ndarray->shape[0]; n++) {
                w = MICROPY_FLOAT_C_FUN(fabs)(array[m*ndarray->shape[0] + n]);
                if((largest < w) && (epsilon < w)) {
                    M = m;
                    N = n;
                    largest = w;
                }
            }
        }
        if(M+N == 0) { // all entries are smaller than epsilon, there is not much we can do...
            break;
        }
        // at this point, we have the pivot, and it is the entry (M, N)
        // now we have to find the rotation angle
        w = (array[N*ndarray->shape[0] + N] - array[M*ndarray->shape[0] + M]) / (2.0*array[M*ndarray->shape[0] + N]);
        // The following if/else chooses the smaller absolute value for the tangent 
        // of the rotation angle. Going with the smaller should be numerically stabler.
        if(w > 0) {
            t = MICROPY_FLOAT_C_FUN(sqrt)(w*w + 1.0) - w;
        } else {
            t = -1.0*(MICROPY_FLOAT_C_FUN(sqrt)(w*w + 1.0) + w);
        }
        s = t / MICROPY_FLOAT_C_FUN(sqrt)(t*t + 1.0); // the sine of the rotation angle
        c = 1.0 / MICROPY_FLOAT_C_FUN(sqrt)(t*t + 1.0); // the cosine of the rotation angle
        tau = (1.0-c)/s; // this is equal to the tangent of the half of the rotation angle
        
        // at this point, we have the rotation angles, so we can transform the matrix
        // first the two diagonal elements
        // a(M, M) = a(M, M) - t*a(M, N)
        array[M*ndarray->shape[0] + M] = array[M*ndarray->shape[0] + M] - t * array[M*ndarray->shape[0] + N];
        // a(N, N) = a(N, N) + t*a(M, N)
        array[N*ndarray->shape[0] + N] = array[N*ndarray->shape[0] + N] + t * array[M*ndarray->shape[0] + N];
        // after the rotation, the a(M, N), and a(N, M) entries should become zero
        array[M*ndarray->shape[0] + N] = array[N*ndarray->shape[0] + M] = 0.0;
        // then all other elements in the column
        for(size_t k=0; k < ndarray->shape[0]; k++) {
            if((k == M) || (k == N)) {
                continue;
            }
            aMk = array[M*ndarray->shape[0] + k];
            aNk = array[N*ndarray->shape[0] + k];
            // a(M, k) = a(M, k) - s*(a(N, k) + tau*a(M, k))
            array[M*ndarray->shape[0] + k] -= s*(aNk + tau*aMk);
            // a(N, k) = a(N, k) + s*(a(M, k) - tau*a(N, k))
            array[N*ndarray->shape[0] + k] += s*(aMk - tau*aNk);
            // a(k, M) = a(M, k)
            array[k*ndarray->shape[0] + M] = array[M*ndarray->shape[0] + k];
            // a(k, N) = a(N, k)
            array[k*ndarray->shape[0] + N] = array[N*ndarray->shape[0] + k];
        }
        // now we have to update the eigenvectors
        // the rotation matrix, R, multiplies from the right
        // R is the unit matrix, except for the 
        // R(M,M) = R(N, N) = c
        // R(N, M) = s
        // (M, N) = -s
        // entries. This means that only the Mth, and Nth columns will change
        for(size_t m=0; m < ndarray->shape[0]; m++) {
            vm = eigvectors[m*ndarray->shape[0]+M];
            vn = eigvectors[m*ndarray->shape[0]+N];
            // the new value of eigvectors(m, M)
            eigvectors[m*ndarray->shape[0]+M] = c * vm - s * vn;
            // the new value of eigvectors(m, N)
            eigvectors[m*ndarray->shape[0]+N] = s * vm + c * vn;
        }
    } while(iterations > 0);
    
    if(iterations == 0) { 
        // the computation did not converge; numpy raises LinAlgError
        m_del(mp_float_t, array, ndarray->len);
        mp_raise_ValueError("iterations did not converge");
    }
    size_t *eigen_shape = m_new(size_t, 1);
    eigen_shape[0] = ndarray->shape[0];
    ndarray_obj_t *eigenvalues = ndarray_new_dense_ndarray(1, eigen_shape, NDARRAY_FLOAT);
    mp_float_t *eigvalues = (mp_float_t *)eigenvalues->array->items;
    for(size_t i=0; i < ndarray->shape[0]; i++) {
        eigvalues[i] = array[i*(ndarray->shape[0]+1)];
    }
    m_del(mp_float_t, array, ndarray->len);
    
    mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(mp_obj_new_tuple(2, NULL));
    tuple->items[0] = MP_OBJ_FROM_PTR(eigenvalues);
    tuple->items[1] = MP_OBJ_FROM_PTR(eigenvectors);
    return tuple;
    return MP_OBJ_FROM_PTR(eigenvalues);
}
