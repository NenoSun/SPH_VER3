// This file sources from https://github.com/finallyjustice/sphfluid


#ifndef __SPHDATA_H__
#define __SPHDATA_H__

#include "Type.cuh"

float window_width = 1000;
float window_height = 1000;

float xRot = 15.0f;
float yRot = 0.0f;
float xTrans = 0.0;
float yTrans = 0;
float zTrans = -35.0;

int ox;
int oy;
int buttonState;
float xRotLength = 0.0f;
float yRotLength = 0.0f;

Float3 real_world_origin;
Float3 real_world_side;
Float3 sim_ratio;

float world_width;
float world_height;
float world_length;

#endif