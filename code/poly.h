/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019-2020 Zoltán Vörös
*/
    
#ifndef _POLY_
#define _POLY_

#include "ulab.h"

#if MICROPY_FLOAT_IMPL == MICROPY_FLOAT_IMPL_FLOAT
#define epsilon        1.2e-7
#elif MICROPY_FLOAT_IMPL == MICROPY_FLOAT_IMPL_DOUBLE
#define epsilon        2.3e-16
#endif

bool poly_invert_matrix(mp_float_t *, size_t );

#if ULAB_POLY_POLYVAL
mp_obj_t poly_polyval(mp_obj_t , mp_obj_t );
MP_DECLARE_CONST_FUN_OBJ_2(poly_polyval_obj);
#endif

#if ULAB_POLY_POLYFIT
mp_obj_t poly_polyfit(size_t  , const mp_obj_t *);
MP_DECLARE_CONST_FUN_OBJ_VAR_BETWEEN(poly_polyfit_obj);
#endif

#endif
