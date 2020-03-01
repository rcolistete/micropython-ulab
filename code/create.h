/*
 * This file is part of the micropython-ulab project,
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 Jeff Epler for Adafruit Industries
 * Copyright (c) 2019-2020 Zoltán Vörös
*/

#ifndef _CREATE_
#define _CREATE_

#include "ulab.h"
#include "ndarray.h"

MP_DECLARE_CONST_FUN_OBJ_KW(create_ones_obj);
MP_DECLARE_CONST_FUN_OBJ_KW(create_zeros_obj);
MP_DECLARE_CONST_FUN_OBJ_KW(create_eye_obj);
MP_DECLARE_CONST_FUN_OBJ_KW(create_linspace_obj);

#endif
