/*
 * This file is part of the micropython-ulab project, 
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019-2020 Zoltán Vörös
*/
    
#ifndef _LINALG_
#define _LINALG_

#include "ulab.h"
#include "ndarray.h"

#if ULAB_LINALG_ZEROS
mp_obj_t linalg_zeros(size_t , const mp_obj_t *, mp_map_t *);
MP_DECLARE_CONST_FUN_OBJ_KW(linalg_zeros_obj);
#endif

#if ULAB_LINALG_ONES
mp_obj_t linalg_ones(size_t , const mp_obj_t *, mp_map_t *);
MP_DECLARE_CONST_FUN_OBJ_KW(linalg_ones_obj);
#endif

#endif
