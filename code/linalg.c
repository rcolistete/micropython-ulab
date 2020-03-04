
/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019-2020 Zoltán Vörös
*/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "py/obj.h"
#include "py/runtime.h"
#include "py/misc.h"
#include "linalg.h"

#if ULAB_LINALG_MODULE
mp_obj_t linalg_size(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = mp_const_none } },
        { MP_QSTR_axis, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = mp_const_none } },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(1, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    if(!MP_OBJ_IS_TYPE(args[0].u_obj, &ulab_ndarray_type)) {
        mp_raise_TypeError(translate("size is defined for ndarrays only"));
    } else {
        ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(args[0].u_obj);
        if(args[1].u_obj == mp_const_none) {
            return mp_obj_new_int(ndarray->len);
        } else if(mp_obj_is_int(args[1].u_obj)) {
            uint8_t ax = mp_obj_get_int(args[1].u_obj);
            if(ax > ndarray->ndim) {
				mp_raise_ValueError("axis index is out of bounds");
			}
			return mp_obj_new_int(ndarray->shape[ax]);
        } else {
            mp_raise_TypeError(translate("wrong argument type"));
        }
    }
}

MP_DEFINE_CONST_FUN_OBJ_KW(linalg_size_obj, 1, linalg_size);
#endif

#if ULAB_LINALG_MODULE || ULAB_POLY_MODULE
bool linalg_invert_matrix(mp_float_t *data, size_t N) {
	// this function is internal, and is not directly exposed to python
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
#endif

#if ULAB_LINALG_MODULE
mp_obj_t linalg_inv(mp_obj_t o_in) {
    // since inv is not a class method, we have to inspect the input argument first
    if(!MP_OBJ_IS_TYPE(o_in, &ulab_ndarray_type)) {
        mp_raise_TypeError(translate("only ndarrays can be inverted"));
    }
    if(!MP_OBJ_IS_TYPE(o_in, &ulab_ndarray_type)) {
        mp_raise_TypeError(translate("only ndarray objects can be inverted"));
    }
	ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(o_in);
    if((ndarray->ndim != 2) || (ndarray->shape[0] != ndarray->shape[1])) {
        mp_raise_ValueError(translate("only square matrices can be inverted"));
    }
    ndarray_obj_t *inverted = ndarray_new_dense_ndarray(2, ndarray->shape, NDARRAY_FLOAT);
    mp_float_t *data = (mp_float_t *)inverted->array;
    mp_obj_t elem;
    for(size_t m=0; m < ndarray->shape[0]; m++) { // rows first
        for(size_t n=0; n < ndarray->shape[1]; n++) { // columns next
            elem = mp_binary_get_val_array(ndarray->dtype, ndarray->array, m*ndarray->shape[0] + ndarray->shape[1]);
            data[m*ndarray->shape[1]+n] = (mp_float_t)mp_obj_get_float(elem);
        }
    }
    
    if(!linalg_invert_matrix(data, ndarray->shape[0])) {
        // TODO: I am not sure this is needed here. Otherwise, 
        // how should we free up the unused RAM of inverted?
        m_del(mp_float_t, inverted->array, ndarray->shape[0]*ndarray->shape[1]);
        mp_raise_ValueError(translate("input matrix is singular"));
    }
    return MP_OBJ_FROM_PTR(inverted);
}

MP_DEFINE_CONST_FUN_OBJ_1(linalg_inv_obj, linalg_inv);

mp_obj_t linalg_dot(mp_obj_t _m1, mp_obj_t _m2) {
    // TODO: should the results be upcast?
    if(!MP_OBJ_IS_TYPE(_m1, &ulab_ndarray_type) || !MP_OBJ_IS_TYPE(_m2, &ulab_ndarray_type)) {
        mp_raise_TypeError(translate("arguments must be ndarrays"));
    }
    ndarray_obj_t *m1 = MP_OBJ_TO_PTR(_m1);
    ndarray_obj_t *m2 = MP_OBJ_TO_PTR(_m2);    
    if(m1->shape[1] != m2->shape[0]) {
        mp_raise_ValueError(translate("matrix dimensions do not match"));
    }
    size_t shape[2] = {m1->shape[0], m2->shape[1]};
    // TODO: numpy uses upcasting here
    ndarray_obj_t *ndarray = ndarray_new_dense_ndarray(2, shape, NDARRAY_FLOAT);
    mp_float_t *array = (mp_float_t *)ndarray->array;
    mp_float_t sum, v1, v2;
    for(size_t i=0; i < m1->shape[0]; i++) { // rows of m1
        for(size_t j=0; j < m2->shape[1]; j++) { // columns of m2
            sum = 0.0;
            for(size_t k=0; k < m2->shape[0]; k++) {
				// TODO: check, whether this works with matrix views
                // (i, k) * (k, j)
                v1 = ndarray_get_float_value(m1->array, m1->dtype, i*m1->shape[0]+k);
                v2 = ndarray_get_float_value(m2->array, m2->dtype, k*m2->shape[1]+j);
                sum += v1 * v2;
            }
            array[j*m1->shape[0]+i] = sum;
        }
    }
    return MP_OBJ_FROM_PTR(ndarray);
}

MP_DEFINE_CONST_FUN_OBJ_2(linalg_dot_obj, linalg_dot);

mp_obj_t linalg_zeros_ones(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args, uint8_t kind) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = MP_OBJ_NULL} } ,
        { MP_QSTR_dtype, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = NDARRAY_FLOAT} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);
    
    uint8_t dtype = args[1].u_int;
    if(!MP_OBJ_IS_INT(args[0].u_obj) && !MP_OBJ_IS_TYPE(args[0].u_obj, &mp_type_tuple)) {
        mp_raise_TypeError(translate("input argument must be an integer or a 2-tuple"));
    }
    ndarray_obj_t *ndarray = NULL;
    if(MP_OBJ_IS_INT(args[0].u_obj)) {
        size_t n = mp_obj_get_int(args[0].u_obj);
        ndarray = ndarray_new_linear_array(n, dtype);
    } else if(MP_OBJ_IS_TYPE(args[0].u_obj, &mp_type_tuple)) {
        ndarray = ndarray_new_ndarray_from_tuple(args[0].u_obj, dtype);
    }
    if(kind == 1) {
        mp_obj_t one = mp_obj_new_int(1);
        for(size_t i=0; i < ndarray->len; i++) {
            mp_binary_set_val_array(dtype, ndarray->array, i, one);
        }
    }
    return MP_OBJ_FROM_PTR(ndarray);
}

mp_obj_t linalg_zeros(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return linalg_zeros_ones(n_args, pos_args, kw_args, 0);
}

MP_DEFINE_CONST_FUN_OBJ_KW(linalg_zeros_obj, 0, linalg_zeros);

mp_obj_t linalg_ones(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    return linalg_zeros_ones(n_args, pos_args, kw_args, 1);
}

MP_DEFINE_CONST_FUN_OBJ_KW(linalg_ones_obj, 0, linalg_ones);

mp_obj_t linalg_eye(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_M, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = mp_const_none } },
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
    size_t shape[2] = {m, n};
    ndarray_obj_t *ndarray = ndarray_new_dense_ndarray(2, shape, dtype);
    mp_obj_t one = mp_obj_new_int(1);
    size_t i = 0;
    if((k >= 0) && (k < n)) {
        while(k < n) {
            mp_binary_set_val_array(dtype, ndarray->array, i*n+k, one);
            k++;
            i++;
        }
    } else if((k < 0) && (-k < m)) {
        k = -k;
        i = 0;
        while(k < m) {
            mp_binary_set_val_array(dtype, ndarray->array, k*n+i, one);
            k++;
            i++;
        }
    }
    return MP_OBJ_FROM_PTR(ndarray);
}

MP_DEFINE_CONST_FUN_OBJ_KW(linalg_eye_obj, 0, linalg_eye);

mp_obj_t linalg_eig(mp_obj_t oin) {
    if(!MP_OBJ_IS_TYPE(oin, &ulab_ndarray_type)) {
        mp_raise_TypeError(translate("function defined for ndarrays only"));
    }
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(oin);
    if((ndarray->ndim != 2) || (ndarray->shape[0] != ndarray->shape[1])) {
        mp_raise_ValueError(translate("input must be square matrix"));
    }
    mp_float_t *array = m_new(mp_float_t, ndarray->len);
    for(size_t m=0; m < ndarray->shape[0]; m++) {
		for(size_t n=0; n < ndarray->shape[0]; n++) {		
			array[m*ndarray->shape[0] + n] = ndarray_get_float_value(ndarray->array, ndarray->dtype, m*ndarray->strides[0] + n*ndarray->strides[1]);
		}
    }
    // make sure the matrix is symmetric
    for(size_t m=0; m < ndarray->shape[0]; m++) {
        for(size_t n=m+1; n < ndarray->shape[1]; n++) {
            // compare entry (m, n) to (n, m)
            // TODO: this must probably be scaled!
            if(epsilon < MICROPY_FLOAT_C_FUN(fabs)(array[m*ndarray->shape[0] + n] - array[n*ndarray->shape[0] + m])) {
                mp_raise_ValueError(translate("input matrix is asymmetric"));
            }
        }
    }
    
    // if we got this far, then the matrix will be symmetric
    
    ndarray_obj_t *eigenvectors = ndarray_new_dense_ndarray(2, ndarray->shape, NDARRAY_FLOAT);
    mp_float_t *eigvectors = (mp_float_t *)eigenvectors->array;
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
            for(size_t n=m+1; n < ndarray->shape[1]; n++) {
                w = MICROPY_FLOAT_C_FUN(fabs)(array[m*ndarray->shape[1] + n]);
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
        w = (array[N*ndarray->shape[1] + N] - array[M*ndarray->shape[1] + M]) / (2.0*array[M*ndarray->shape[1] + N]);
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
        array[M*ndarray->shape[1] + M] = array[M*ndarray->shape[1] + M] - t * array[M*ndarray->shape[1] + N];
        // a(N, N) = a(N, N) + t*a(M, N)
        array[N*ndarray->shape[1] + N] = array[N*ndarray->shape[1] + N] + t * array[M*ndarray->shape[1] + N];
        // after the rotation, the a(M, N), and a(N, M) entries should become zero
        array[M*ndarray->shape[1] + N] = array[N*ndarray->shape[1] + M] = 0.0;
        // then all other elements in the column
        for(size_t k=0; k < ndarray->shape[0]; k++) {
            if((k == M) || (k == N)) {
                continue;
            }
            aMk = array[M*ndarray->shape[1] + k];
            aNk = array[N*ndarray->shape[1] + k];
            // a(M, k) = a(M, k) - s*(a(N, k) + tau*a(M, k))
            array[M*ndarray->shape[1] + k] -= s*(aNk + tau*aMk);
            // a(N, k) = a(N, k) + s*(a(M, k) - tau*a(N, k))
            array[N*ndarray->shape[1] + k] += s*(aMk - tau*aNk);
            // a(k, M) = a(M, k)
            array[k*ndarray->shape[1] + M] = array[M*ndarray->shape[1] + k];
            // a(k, N) = a(N, k)
            array[k*ndarray->shape[1] + N] = array[N*ndarray->shape[1] + k];
        }
        // now we have to update the eigenvectors
        // the rotation matrix, R, multiplies from the right
        // R is the unit matrix, except for the 
        // R(M,M) = R(N, N) = c
        // R(N, M) = s
        // (M, N) = -s
        // entries. This means that only the Mth, and Nth columns will change
        for(size_t m=0; m < ndarray->shape[1]; m++) {
            vm = eigvectors[m*ndarray->shape[1]+M];
            vn = eigvectors[m*ndarray->shape[1]+N];
            // the new value of eigvectors(m, M)
            eigvectors[m*ndarray->shape[1]+M] = c * vm - s * vn;
            // the new value of eigvectors(m, N)
            eigvectors[m*ndarray->shape[1]+N] = s * vm + c * vn;
        }
    } while(iterations > 0);
    
    if(iterations == 0) { 
        // the computation did not converge; numpy raises LinAlgError
        m_del(mp_float_t, array, ndarray->len);
        mp_raise_ValueError(translate("iterations did not converge"));
    }
    ndarray_obj_t *eigenvalues = ndarray_new_linear_array(ndarray->shape[1], NDARRAY_FLOAT);
    mp_float_t *eigvalues = (mp_float_t *)eigenvalues->array;
    for(size_t i=0; i < ndarray->shape[1]; i++) {
        eigvalues[i] = array[i*(ndarray->shape[1]+1)];
    }
    m_del(mp_float_t, array, ndarray->len);
    
    mp_obj_tuple_t *tuple = MP_OBJ_TO_PTR(mp_obj_new_tuple(2, NULL));
    tuple->items[0] = MP_OBJ_FROM_PTR(eigenvalues);
    tuple->items[1] = MP_OBJ_FROM_PTR(eigenvectors);
    return tuple;
    return MP_OBJ_FROM_PTR(eigenvalues);
}

MP_DEFINE_CONST_FUN_OBJ_1(linalg_eig_obj, linalg_eig);

mp_obj_t linalg_cholesky(mp_obj_t oin) {
	if(!MP_OBJ_IS_TYPE(oin, &ulab_ndarray_type)) {
		mp_raise_TypeError(translate("function is defined for ndarrays only"));
	}
	ndarray_obj_t *in = MP_OBJ_TO_PTR(oin);
	if((in->ndim != 2) || (in->shape[0] != in->shape[1])) {
		mp_raise_ValueError(translate("input must be square matrix"));
	}
	return mp_const_none;
}

MP_DEFINE_CONST_FUN_OBJ_1(linalg_cholesky_obj, linalg_cholesky);


STATIC const mp_rom_map_elem_t ulab_linalg_globals_table[] = {
    { MP_OBJ_NEW_QSTR(MP_QSTR___name__), MP_OBJ_NEW_QSTR(MP_QSTR_linalg) },
    { MP_ROM_QSTR(MP_QSTR_size), (mp_obj_t)&linalg_size_obj },
    { MP_ROM_QSTR(MP_QSTR_inv), (mp_obj_t)&linalg_inv_obj },
    { MP_ROM_QSTR(MP_QSTR_dot), (mp_obj_t)&linalg_dot_obj },
    { MP_ROM_QSTR(MP_QSTR_zeros), (mp_obj_t)&linalg_zeros_obj },
    { MP_ROM_QSTR(MP_QSTR_ones), (mp_obj_t)&linalg_ones_obj },
    { MP_ROM_QSTR(MP_QSTR_eye), (mp_obj_t)&linalg_eye_obj },
    { MP_ROM_QSTR(MP_QSTR_eig), (mp_obj_t)&linalg_eig_obj },
	{ MP_ROM_QSTR(MP_QSTR_cholesky), (mp_obj_t)&linalg_cholesky_obj },
};

STATIC MP_DEFINE_CONST_DICT(mp_module_ulab_linalg_globals, ulab_linalg_globals_table);

mp_obj_module_t ulab_linalg_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&mp_module_ulab_linalg_globals,
};

#endif
