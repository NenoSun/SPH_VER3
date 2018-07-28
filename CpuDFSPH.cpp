#include "CpuDFSPH.h"
#ifdef CPU_DF
#include <iostream>
#define print(x) std::cout << x << std::endl;


int Cpu_device_cube_edge_flags[256] =
{
	0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

int Cpu_device_triangle_table[256][16] =
{
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1 },
	{ 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1 },
	{ 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1 },
	{ 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1 },
	{ 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1 },
	{ 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 },
	{ 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1 },
	{ 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 },
	{ 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1 },
	{ 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1 },
	{ 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1 },
	{ 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1 },
	{ 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 },
	{ 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 },
	{ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1 },
	{ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 },
	{ 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1 },
	{ 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1 },
	{ 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1 },
	{ 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1 },
	{ 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1 },
	{ 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1 },
	{ 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1 },
	{ 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1 },
	{ 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1 },
	{ 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1 },
	{ 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1 },
	{ 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1 },
	{ 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1 },
	{ 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 },
	{ 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1 },
	{ 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1 },
	{ 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1 },
	{ 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1 },
	{ 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1 },
	{ 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 },
	{ 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 },
	{ 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1 },
	{ 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1 },
	{ 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1 },
	{ 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1 },
	{ 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 },
	{ 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 },
	{ 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1 },
	{ 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1 },
	{ 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1 },
	{ 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1 },
	{ 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 },
	{ 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1 },
	{ 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1 },
	{ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1 },
	{ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1 },
	{ 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1 },
	{ 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1 },
	{ 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1 },
	{ 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1 },
	{ 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1 },
	{ 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1 },
	{ 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1 },
	{ 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1 },
	{ 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1 },
	{ 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1 },
	{ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1 },
	{ 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1 },
	{ 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1 },
	{ 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1 },
	{ 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 },
	{ 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 },
	{ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1 },
	{ 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1 },
	{ 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1 },
	{ 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1 },
	{ 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1 },
	{ 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1 },
	{ 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1 },
	{ 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1 },
	{ 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1 },
	{ 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1 },
	{ 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1 },
	{ 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1 },
	{ 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1 },
	{ 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1 },
	{ 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1 },
	{ 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1 },
	{ 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1 },
	{ 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1 },
	{ 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1 },
	{ 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1 },
	{ 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1 },
	{ 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1 },
	{ 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1 },
	{ 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 },
	{ 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }
};

int Cpu_device_edge_conn[12][2] =
{
	{ 0,1 },{ 1,2 },{ 2,3 },{ 3,0 },
	{ 4,5 },{ 5,6 },{ 6,7 },{ 7,4 },
	{ 0,4 },{ 1,5 },{ 2,6 },{ 3,7 }
};

Uint3 Cpu_computeCellPosition(Float3 pos, Param* param) {
	Uint3 cellPos;
	cellPos.x = (uint)floor(pos.x / param->h);
	cellPos.y = (uint)floor(pos.y / param->h);
	cellPos.z = (uint)floor(pos.z / param->h);
	return cellPos;
}

uint Cpu_computeCellHash(Uint3 cellPos, Param* param) {
	if (cellPos.x>param->gridSize.x - 1 || cellPos.y>param->gridSize.y - 1 || cellPos.z>param->gridSize.z - 1)
		return -1;

	return (uint)(cellPos.z*param->gridSize.x*param->gridSize.y + cellPos.y*param->gridSize.x + cellPos.x);
}

void Cpu_DFSPHSetUp(Particle* dParticles, Param* param, uint* dParticleIndex, uint* dCellIndex, uint* dStart, uint* dEnd, cube* dCubes, Float3* dTriangles, Param* hParam,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd) {
	for (int i = 0; i < param->num_particles; i++) {
		dCellIndex[i] = 0xffffffff;
		dParticleIndex[i] = 0xffffffff;
	}
	for (int i = 0; i < param->cells_total; i++) {
		dStart[i] = 0xffffffff;
		dEnd[i] = 0xffffffff;
	}
	Cpu_generateHashTable(dParticles, dParticleIndex, dCellIndex, param);
	Cpu_sort_particles(dCellIndex, dParticleIndex, param->num_particles);
	Cpu_find_start_end_kernel(dStart, dEnd, dCellIndex, dParticleIndex, param->num_particles);
	Cpu_find_start_end_kernel(dStart, dEnd, dCellIndex, dParticleIndex, param->num_particles);
	Cpu_DFSPHCommputeDensityAndFactorAlpha(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd);
}

void Cpu_DFSPHLoop(Particle* dParticles, Param* param, uint* dParticleIndex, uint* dCellIndex, uint* dStart, uint* dEnd, cube* Cubes, Float3* Triangles, Param* Param,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd) {
	Cpu_DFSPHComputeNormals(dParticles, param, dStart, dEnd, dParticleIndex);
#ifdef VISCOUS_FORCE
	Cpu_DFSPHComputeForces(dParticles, param, dStart, dEnd, dParticleIndex);
#endif
#ifdef SURFACE_TENSION
	Cpu_DFSPHComputeSurfaceTensionForce(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd);
#endif SURFACE_TENSION
	Cpu_DFSPHComputeVelocityScalar(dParticles, param);
	Cpu_DFSPHUpdateTimeStep(dParticles, param);
	Cpu_DFSPHPredictVelocity(dParticles, param);
	int counter = 0;
	int dIsGood = 0;
#ifdef DENSITY_SOLVER
	while ((dIsGood == 0 || counter < 2) && counter < 100) {
		param->avg_dens = 0.0f;
		Cpu_DFSPHPredictDensAndVelocity(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd, &dIsGood);
		Cpu_DFSPHDensitySolverPart3(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd, &dIsGood);
		Cpu_DFSPHDensitySolverPart2(param, &dIsGood);
		printf("DENSITY SOLVER ITERATION %d\n", counter);
		counter++;
	}
#endif
	Cpu_DFSPHUpdatePosition(dParticles, param);

	for (int i = 0; i < param->num_particles; i++) {
		dCellIndex[i] = 0xffffffff;
		dParticleIndex[i] = 0xffffffff;
	}
	for (int i = 0; i < param->cells_total; i++) {
		dStart[i] = 0xffffffff;
		dEnd[i] = 0xffffffff;
	}
	Cpu_generateHashTable(dParticles, dParticleIndex, dCellIndex, param);
	Cpu_sort_particles(dCellIndex, dParticleIndex, param->num_particles);
	Cpu_find_start_end_kernel(dStart, dEnd, dCellIndex, dParticleIndex, param->num_particles);
	Cpu_DFSPHCommputeDensityAndFactorAlpha(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd);

#ifdef DIVERGENCE_SOLVER
	dIsGood = 0;
	counter = 0;
	while ((dIsGood == 0 || counter < 1) && counter < 100) {
		param->avg_grad_dens = 0.0f;
		Cpu_DFSPHDivergenceSolver1(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd, &dIsGood);
		Cpu_DFSPHDivergenceSolver3(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd, &dIsGood);
		Cpu_DFSPHDivergenceSolver2(param, &dIsGood);
		printf("DIVERGENCE SOLVER ITERATION %d\n", counter);
		counter++;
	}
#endif
}

 void Cpu_DFSPHComputeNormals(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex) {
	for (int index = 0; index < param->num_particles; index++) {
		Uint3 neighborPos;
		Uint3 cellPos = Cpu_computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);
		p->norm.x = p->norm.y = p->norm.z = 0.0f;
		uint count = 0;

		ITERATE_NEIGHBOR{
		neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		int hash = Cpu_computeCellHash(neighborPos, param);
		if (hash < 0 || hash >= param->cells_total)
			continue;
		// If there exists particles in the cell_index
		if (dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *current = &particles[dParticleIndex[count]];
				float distance = current->pos.cpu_Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Particle j = (*current);
				Float3 deltaR = p->pos - j.pos;

				if (q <= 0.5) {
					p->norm += param->mass * param->grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR / distance / j.dens;
				}

				else if (q <= 1) {
					p->norm += param->mass * param->grad_spline_coff * (-1) * pow(1.0f - q, 2) * deltaR / distance / j.dens;
				}
			}
		}
		}

		p->norm = param->h * p->norm;
	}
}

 void Cpu_generateHashTable(Particle* particles, uint* dParticleIndex, uint* dCellIndex, Param* param) {
	for (int index = 0; index < param->num_particles; index++) {
		// Compute the cell index
		uint hash = Cpu_computeCellHash(Cpu_computeCellPosition(particles[index].pos, param), param);
		if (hash >= param->cells_total)
			continue;
		dParticleIndex[index] = index;
		dCellIndex[index] = hash;
	}
}

 void Cpu_sort_particles(uint *dHash, uint *dIndex, int num_particle) {
	if (num_particle == 0)
		return;

	struct Interval
	{
		int left = 0;
		int right = 0;

		Interval(int newLeft, int newRight) {
			left = newLeft;
			right = newRight;
		}

		int getMiddle() {
			return (left + right) / 2;
		}
	};

	std::vector<Interval> stack;
	std::vector<uint> smallHashStack;
	std::vector<uint> bigHashStack;
	std::vector<uint> smallIndexStack;
	std::vector<uint> bigIndexStack;

	Interval interval(0, num_particle);
	Interval *current;

	stack.push_back(interval);


	while (stack.size() != 0) {
		current = &stack[stack.size() - 1];
		if (current->right - current->left <= 1) {
			stack.pop_back();
			continue;
		}
		int pivot_index = current->getMiddle();
		uint pivot = dHash[pivot_index];
		uint pivot_dIndex = dIndex[pivot_index];
		for (int i = current->left; i < current->right; i++) {
			if (i == pivot_index)
				continue;
			if (dHash[i] < pivot) {
				smallHashStack.push_back(dHash[i]);
				smallIndexStack.push_back(dIndex[i]);
			}
			else {
				bigHashStack.push_back(dHash[i]);
				bigIndexStack.push_back(dIndex[i]);
			}
		}
		int j = current->left;
		for (int k = 0; k < smallHashStack.size(); k++) {
			dHash[j] = smallHashStack[k];
			dIndex[j] = smallIndexStack[k];
			j++;
		}
		dHash[j] = pivot;
		dIndex[j] = pivot_dIndex;
		pivot_index = j;
		j++;
		for (int k = 0; k < bigHashStack.size(); k++) {
			dHash[j] = bigHashStack[k];
			dIndex[j] = bigIndexStack[k];
			j++;
		}
		smallHashStack.clear();
		smallIndexStack.clear();
		bigHashStack.clear();
		bigIndexStack.clear();
		stack.pop_back();
		Interval leftSegment(current->left, pivot_index);
		Interval rightSegment(pivot_index + 1, current->right);
		stack.push_back(leftSegment);
		stack.push_back(rightSegment);
	}
}

 void Cpu_DFSPHComputeForces(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex) {
	// Suppose now we've got the external forces
	for (int index = 0; index < param->num_particles; index++) {
		Uint3 neighborPos;
		Uint3 cellPos = Cpu_computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);
		p->acc.x = p->acc.z = 0;
		p->acc.y = 0;
		uint count = 0;

		ITERATE_NEIGHBOR{
		neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		int hash = Cpu_computeCellHash(neighborPos, param);
		if (hash<0 || hash >= param->cells_total)
			continue;
		if (dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *current = &particles[dParticleIndex[count]];
				float distance = current->pos.cpu_Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;
				// To optimize the calculation process
				Particle j = (*current);
				Float3 delta_v = p->vel - j.vel;

				if (q <= 0.5) {
					p->acc -= VISCOUS_FROCE_COEFFICIENT * param->vicosity_coff * (param->mass / j.dens) * delta_v * param->spline_coff * (6 * pow(q, 3) - 6 * pow(q, 2) + 1) / param->timeStep;
				}

				else if (q <= 1) {
					p->acc -= VISCOUS_FROCE_COEFFICIENT * param->vicosity_coff * (param->mass / j.dens) * delta_v * param->spline_coff * 2 * pow(1 - q, 3) / param->timeStep;
				}
			}
		}
		}
	}
}

 void Cpu_DFSPHComputeSurfaceTensionForce(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd) {
	for (int index = 0; index < param->num_particles; index++) {
		int hash;
		Uint3 neighborPos;
		Uint3 cellPos = Cpu_computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);
		uint count = 0;

		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = Cpu_computeCellHash(neighborPos, param);
		if (hash < 0 || hash >= param->cells_total)
			continue;
		if (dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *current = &particles[dParticleIndex[count]];
				float distance = current->pos.cpu_Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;
				// To optimize the calculation process
				Particle j = (*current);
				float K_ij = 2.0 * param->rest_density / (p->dens + j.dens);
				Float3 temp;
				temp.x = temp.y = temp.z = 0.0f;
				Float3 deltaR = p->pos - j.pos;

				if (deltaR.NormSquare() > 1.0e-9) {
					deltaR = deltaR * (1.0 / deltaR.Norm());
					if (q > 0.5)
						temp -= param->surf_tens_coff * param->mass * deltaR * param->cohesion_coff * pow(param->h - distance, 3) * pow(distance, 3);
					else
						temp -= param->surf_tens_coff * param->mass * deltaR * param->cohesion_coff * 2.0f * pow(param->h - distance, 3) * pow(distance, 3) - param->cohesion_term;
				}

				temp -= param->surf_tens_coff * param->h * (p->norm - j.norm);

				p->acc += SURFACE_TENSION_COEFFICIENT * K_ij * temp;
			}
		}
#ifdef ENABLE_BOUNDARY_PARTICLE
		if (dBoundaryStart[hash] < param->num_boundary_particles) {
			for (count = dBoundaryStart[hash]; count <= dBoundaryEnd[hash]; count++) {
				Particle *j = &dBoundaryParticles[dBoundaryParticleIndex[count]];
				float distance = j->pos.Dist(p->pos);
				float distanceSquare = distance * distance;
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;

				if (deltaR.NormSquare() > 1.0e-9) {
					deltaR = (1.0 / deltaR.Norm()) * deltaR;
					if (q > 0.5)
						p->acc -= SURFACE_TENSION_COEFFICIENT * param->surf_tens_coff * j->Psi * deltaR * param->Adhesion_coff * pow(-4.0 * distanceSquare / param->h + 6.0f*distance - 2.0f*param->h, 0.25);
				}
			}
		}
#endif
		}
	}
}

 void Cpu_DFSPHComputeVelocityScalar(Particle* particles, Param* param) {
	for (int index = 0; index < param->num_particles; index++) {
		Particle *p = &particles[index];
		p->vel_scalar = p->vel.Norm();
	}
}

 void Cpu_DFSPHUpdateTimeStep(Particle* particles, Param* param) {
	float max_vel = particles[0].vel_scalar;
	for (int i = 1; i < param->num_particles; i++) {
		if (particles[i].vel_scalar > max_vel)
			max_vel = particles[i].vel_scalar;
	}

	// Use the middle value
	param->timeStep = 0.5 * 0.4 * 2 * param->radius / (max_vel + 1e-6);

	if (param->timeStep > 0.005)
		param->timeStep = 0.005;

	else if (param->timeStep < 0.0001)
		param->timeStep = 0.0001;

}

 void Cpu_DFSPHPredictVelocity(Particle* particles, Param* param) {
	for (int index = 0; index < param->num_particles; index++) {
		Particle *p = &particles[index];

		p->vel.x += param->timeStep * p->acc.x;
		p->vel.y += param->timeStep * ( p->acc.y + GRAVITY) ;
		p->vel.z += param->timeStep * p->acc.z;
	}
}

 void Cpu_DFSPHPredictDensAndVelocity(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood) {
	for (int index = 0; index < param->num_particles; index++) {
		uint hash;
		uint count = 0;
		Uint3 neighborPos;
		Uint3 cellPos = Cpu_computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);

		p->predict_dens = 0;

		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = Cpu_computeCellHash(neighborPos, param);
		if (hash >= param->cells_total)
			continue;

		if (dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *j = &particles[dParticleIndex[count]];
				float distance = j->pos.cpu_Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;
				Float3 deltaV = p->vel - j->vel;

				// Compute Density
				if (q <= 0.5)
					p->predict_dens += param->mass * param->grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR.Dot(deltaV) / distance;
				else if (q <= 1)
					p->predict_dens += param->mass * param->grad_spline_coff * (-1) * pow(1.0f - q, 2) * deltaR.Dot(deltaV) / distance;
			}
		}
#ifdef ENABLE_BOUNDARY_PARTICLE

		if (dBoundaryStart[hash] < param->num_boundary_particles) {
			for (count = dBoundaryStart[hash]; count <= dBoundaryEnd[hash]; count++) {
				Particle *j = &dBoundaryParticles[dBoundaryParticleIndex[count]];
				float distance = j->pos.Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;
				Float3 deltaV = p->vel - j->vel;

				// Compute Density
				if (q <= 0.5)
					p->predict_dens += j->Psi * param->grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR.Dot(deltaV) / distance;
				else if (q <= 1)
					p->predict_dens += j->Psi * param->grad_spline_coff * (-1) * pow(1.0f - q, 2) * deltaR.Dot(deltaV) / distance;
			}
		}
#endif
		}

		p->predict_dens = p->dens + param->timeStep * p->predict_dens;

		if (p->predict_dens < param->rest_density)
			p->predict_dens = param->rest_density;

		// CUDA calculation accruary issue
		float dens_err = p->predict_dens - param->rest_density;
		if (dens_err > 1e-6)
			param->avg_dens += dens_err;
	}
}

 void Cpu_DFSPHDensitySolverPart3(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood) {

	for (int index = 0; index < param->num_particles; index++) {
		uint hash;
		uint count = 0;
		Uint3 neighborPos;
		Uint3 cellPos = Cpu_computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);

		float ki = (p->predict_dens - param->rest_density) * p->alpha / (param->timeStep*param->timeStep);


		float kj = 0;
		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = Cpu_computeCellHash(neighborPos, param);
		if (hash >= param->cells_total)
			continue;
		if (dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *j = &particles[dParticleIndex[count]];
				float distance = j->pos.cpu_Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;
				kj = (j->predict_dens - param->rest_density) * j->alpha / (param->timeStep*param->timeStep);

				// Prevent instability
				float tk = ki + kj;
				if (-1e-6 >= tk || tk >= 1e-6) {
					if (q <= 0.5) {
						p->vel += SURFACE_TENSION_COEFFICIENT * param->timeStep * tk * param->mass * param->grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR / distance;
					}
					else if (q <= 1) {
						p->vel += SURFACE_TENSION_COEFFICIENT * param->timeStep * tk * param->mass * param->grad_spline_coff * (-1) * pow(1.0f - q, 2) * deltaR / distance;
					}
				}
			}
		}
#ifdef ENABLE_BOUNDARY_PARTICLE

		if (dBoundaryStart[hash] < param->num_boundary_particles) {
			for (count = dBoundaryStart[hash]; count <= dBoundaryEnd[hash]; count++) {
				Particle *j = &dBoundaryParticles[dBoundaryParticleIndex[count]];
				float distance = j->pos.Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;

				if (ki <= -1e-6 || ki >= 1e-6) {
					if (q <= 0.5) {
						p->vel += SURFACE_TENSION_COEFFICIENT * param->timeStep * j->Psi * ki * param->grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR / distance;
					}

					else if (q <= 1) {
						p->vel += SURFACE_TENSION_COEFFICIENT * param->timeStep * j->Psi * ki * param->grad_spline_coff * (-1) * pow(1.0f - q, 2) * deltaR / distance;
					}
				}
			}
		}
#endif
		}
	}
}

 void Cpu_DFSPHDensitySolverPart2(Param* param, int *isGood) {
	if (-0.1 < param->avg_dens && param->avg_dens < 0.1) {
		*isGood = 1;
		return;
	}
	param->avg_dens = param->avg_dens / param->num_particles;
	if (-0.1 < param->avg_dens && param->avg_dens < 0.1)
		*isGood = 1;
}

void Cpu_find_start_end_kernel(uint *dStart, uint *dEnd, uint *dCellIndex, uint *dParticleIndex, uint num_particle)
{
	// For each index in the dParticleIndex
	for (int index = 0; index < num_particle; index++) {
		// If index == 0
		if (index == 0) {
			// Then we check if we have any particle
			if (dCellIndex[0] == 0xffffffff && dCellIndex[1] == 0xffffffff)
				continue;
			else if (dCellIndex[1] == 0xffffffff) {
				dStart[dCellIndex[0]] = index;
				dEnd[dCellIndex[0]] = index;
				continue;
			}
			else if (dCellIndex[index] == dCellIndex[index + 1])
				dStart[dCellIndex[index]] = index;
			else {
				dStart[dCellIndex[index]] = dEnd[dCellIndex[index]] = index;
				dStart[dCellIndex[index + 1]] = index + 1;
			}
		}

		else if (index == num_particle - 1) {
			if (dCellIndex[index] == 0xffffffff && dCellIndex[index - 1] == 0xffffffff)
				continue;
			else if (dCellIndex[index] == 0xffffffff) {
				dEnd[dCellIndex[index - 1]] = index - 1;
			}
			else if (dCellIndex[index] == dCellIndex[index - 1])
				dEnd[dCellIndex[index]] = index;
			else {
				dStart[dCellIndex[index]] = index;
				dEnd[dCellIndex[index]] = index;
			}
		}

		else if (dCellIndex[index] == dCellIndex[index + 1]) {
			continue;
		}

		else {
			if (dCellIndex[index] != 0xffffffff && dCellIndex[index + 1] == 0xffffffff) {
				dEnd[dCellIndex[index]] = index;
			}
			else {
				dEnd[dCellIndex[index]] = index;
				dStart[dCellIndex[index + 1]] = index + 1;
			}
		}
	}
}

 void Cpu_DFSPHCommputeDensityAndFactorAlpha(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd) {
	for (int index = 0; index < param->num_particles; index++)
	{
		uint hash;
		uint count = 0;
		Uint3 neighborPos;
		Uint3 cellPos = Cpu_computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);
		p->dens = param->mass * param->spline_coff;
		p->alpha = 0;
		Float3 temp, temp2;
		temp.x = temp.y = temp.z = 0;
		temp2.x = temp2.y = temp2.z = 0;

		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = Cpu_computeCellHash(neighborPos, param);
		if (hash >= param->cells_total)
			continue;

		if (dStart[hash] >= 0 && dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *j = &particles[dParticleIndex[count]];
				float distance = j->pos.cpu_Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;

				// Compute Density
				if (q <= 0.5)
					p->dens += param->mass * param->spline_coff * (6 * pow(q, 3) - 6 * pow(q, 2) + 1);
				else if (q <= 1)
					p->dens += param->mass * param->spline_coff * 2 * pow(1 - q, 3);

				// Compute Factor Alpha
				if (q <= 0.5) {
					temp2 = param->mass * param->grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR / distance;
					p->alpha += temp2.NormSquare();
					temp += temp2;
				}

				else if (q <= 1) {
					temp2 = param->mass * param->grad_spline_coff * (-1) * pow(1.0f - q, 2) * deltaR / distance;
					p->alpha += temp2.NormSquare();
					temp += temp2;
				}
			}
		}

#ifdef ENABLE_BOUNDARY_PARTICLE
		if (dBoundaryStart[hash] < 0 || dBoundaryEnd[hash] >= param->num_boundary_particles)
			continue;
		for (count = dBoundaryStart[hash]; count <= dBoundaryEnd[hash]; count++) {
			Particle *j = &dBoundaryParticles[dBoundaryParticleIndex[count]];
			float distance = j->pos.Dist(p->pos);
			float q = distance / param->h;
			if (q > 1 || q <= 0)
				continue;

			Float3 deltaR = p->pos - j->pos;

			// Compute Density
			if (q <= 0.5)
				p->dens += j->Psi * param->spline_coff * (6 * pow(q, 3) - 6 * pow(q, 2) + 1);
			else if (q <= 1)
				p->dens += j->Psi * param->spline_coff * 2 * pow(1 - q, 3);

			// Compute Factor Alpha 
			if (q <= 0.5) {
				temp2 = j->Psi * param->grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR / distance;
				p->alpha += temp2.NormSquare();
				temp += temp2;
			}
			else if (q <= 1) {
				temp2 = j->Psi * param->grad_spline_coff * (-1) * pow(1.0f - q, 2) * deltaR / distance;
				p->alpha += temp2.NormSquare();
				temp += temp2;
			}
		}
#endif
		}


		p->alpha += temp.NormSquare();

		if (p->alpha < 1e-6)
			p->alpha = 1e-6;

		p->alpha = -1.0f / p->alpha;

	}

}

 void Cpu_DFSPHDivergenceSolver1(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood) {

	for (int index = 0; index < param->num_particles; index++) {
		int hash;
		uint count = 0;
		Uint3 neighborPos;
		Uint3 cellPos = Cpu_computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);
		p->grad_dens = 0.0f;
		int neighbor_count = 0;

		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = Cpu_computeCellHash(neighborPos, param);
		if (hash < 0 || hash >= param->cells_total)
			continue;
		if (dStart[hash] >= 0 && dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *j = &particles[dParticleIndex[count]];
				float distance = j->pos.cpu_Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;
				Float3 deltaV = p->vel - j->vel;

				// Compute Density
				if (q <= 0.5)
					p->grad_dens += param->mass  * param->grad_spline_coff * q * (3.0f*q - 2.0f) * (deltaR.Dot(deltaV)) / distance;
				else if (q <= 1)
					p->grad_dens += param->mass * param->grad_spline_coff * (-1) * pow(1 - q, 2) * (deltaR.Dot(deltaV)) / distance;
			}
		}
#ifdef ENABLE_BOUNDARY_PARTICLE
		if (dBoundaryStart[hash] < 0 || dBoundaryStart[hash] >= param->num_boundary_particles)
			continue;
		for (count = dBoundaryStart[hash]; count <= dBoundaryEnd[hash]; count++) {
			Particle *j = &dBoundaryParticles[dBoundaryParticleIndex[count]];
			float distance = j->pos.cpu_Dist(p->pos);
			float q = distance / param->h;
			if (q > 1 || q <= 0)
				continue;

			Float3 deltaR = p->pos - j->pos;
			Float3 deltaV = p->vel - j->vel;
			neighbor_count++;

			// Compute Density Gradient
			if (q <= 0.5)
				p->grad_dens += j->Psi  * param->grad_spline_coff * q * (3.0f*q - 2.0f) * (deltaR.Dot(deltaV)) / distance;
			else if (q <= 1)
				p->grad_dens += j->Psi * param->grad_spline_coff * (-1) * pow(1 - q, 2) *(deltaV.Dot(deltaR)) / distance;
		}
#endif
		}


	    if (p->grad_dens < 0)
	    	p->grad_dens = 0;
		if (neighbor_count < 20)
			p->grad_dens = 0;

		// CUDA calculation accruary issue
		if (p->grad_dens > 1e-6)
			param->avg_grad_dens += p->grad_dens;
	}
}

 void Cpu_DFSPHDivergenceSolver3(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood) {
	for (int index = 0; index < param->num_particles; index++) {
		int hash;
		uint count = 0;
		Uint3 neighborPos;
		Uint3 cellPos = Cpu_computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);

		float ki = p->grad_dens * p->alpha / param->timeStep;
		float kj = 0;
		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = Cpu_computeCellHash(neighborPos, param);
		if (hash < 0 || hash >= param->cells_total)
			continue;
		if (dStart[hash] >= 0 && dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *j = &particles[dParticleIndex[count]];
				float distance = j->pos.cpu_Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;
				kj = j->grad_dens * j->alpha / param->timeStep;

				// Compute Density
				float tk = ki + kj;
				if (tk < -1e-6 || tk > 1e-6) {
					if (q <= 0.5) {
						p->vel += SURFACE_TENSION_COEFFICIENT * param->timeStep * param->mass * param->grad_spline_coff*(ki + kj) * q * (3.0f*q - 2.0f) * deltaR / distance;
					}
					else if (q <= 1) {
						p->vel += SURFACE_TENSION_COEFFICIENT * param->timeStep*param->mass * param->grad_spline_coff * (ki + kj) * (-1) * pow(1 - q, 2) * deltaR / distance;
					}
				}
			}
		}

#ifdef ENABLE_BOUNDARY_PARTICLE
		if (dBoundaryStart[hash] < 0 || dBoundaryStart[hash] >= param->num_boundary_particles)
			continue;
		for (count = dBoundaryStart[hash]; count <= dBoundaryEnd[hash]; count++) {
			Particle *j = &dBoundaryParticles[dBoundaryParticleIndex[count]];
			float distance = j->pos.cpu_Dist(p->pos);
			float q = distance / param->h;
			if (q > 1 || q <= 0)
				continue;

			Float3 deltaR = p->pos - j->pos;

			// Compute Density
			if (ki < -1e-6 || ki > 1e-6) {
				if (q <= 0.5) {
					p->vel += SURFACE_TENSION_COEFFICIENT * param->timeStep * j->Psi * param->grad_spline_coff * (ki)* q * (3.0f*q - 2.0f) * deltaR / distance;
				}
				else if (q <= 1) {
					p->vel += SURFACE_TENSION_COEFFICIENT *param->timeStep * j->Psi * param->grad_spline_coff * (ki) * (-1) * pow(1 - q, 2) * deltaR / distance;
				}
			}
		}
#endif
		}
	}
}

 void Cpu_DFSPHDivergenceSolver2(Param* param, int *isGood) {
	param->avg_grad_dens = param->avg_grad_dens / param->num_particles;
	if (param->avg_grad_dens < (1.0f / param->timeStep))
		*isGood = 1;
}

 void Cpu_DFSPHUpdatePosition(Particle* particles, Param* param) {
	for (int index = 0; index < param->num_particles; index++) {
		Particle *p = &particles[index];
		p->pos += param->timeStep * ( p->vel );
	}
}

 void generateHashTable_Boundary(Particle* particles, uint* dParticleIndex, uint* dCellIndex, Param* param) {
	// Each thread represents for a particle
	for (int index = 0; index < param->num_boundary_particles; index++) {
		// Compute the cell index
		uint hash = Cpu_computeCellHash(Cpu_computeCellPosition(particles[index].pos, param), param);
		if (hash == -1)
			continue;
		dParticleIndex[index] = index;
		dCellIndex[index] = hash;
	}
}

 void Cpu_computeBorderPsi(Particle* dParticle, Param* param, uint* dParticleIndex, uint* dCellIndex, uint* dStart, uint* dEnd) {
	for (int index = 0; index < param->num_boundary_particles; index++) {
		uint hash = Cpu_computeCellHash(Cpu_computeCellPosition(dParticle[index].pos, param), param);
		if (hash >= param->cells_total)
			continue;
		Uint3 neighborPos;
		Uint3 cellPos = Cpu_computeCellPosition(dParticle[index].pos, param);
		Particle *p = &(dParticle[index]);
		p->Psi = 0;

		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = Cpu_computeCellHash(neighborPos, param);
		if (hash >= param->cells_total)
			continue;
		if (dStart[hash] < 0 || dStart[hash] >= param->num_boundary_particles)
			continue;
		uint count = 0;
		for (count = dStart[hash]; count <= dEnd[hash]; count++) {
			Particle *j = &dParticle[dParticleIndex[count]];
			float distance = j->pos.Dist(p->pos);
			float q = distance / param->h;
			if (q > 1 || q <= 0)
				continue;

			Float3 deltaR = p->pos - j->pos;

			// Compute Density
			if (q <= 0.5)
				p->Psi += param->spline_coff * (6 * pow(q, 3) - 6 * pow(q, 2) + 1);
			else if (q <= 1)
				p->Psi += param->spline_coff * 2 * pow(1 - q, 3);
		}
		}
		p->Psi = param->rest_density / p->Psi;
	}
}

float Cpu_dCalDistance(Float3 p1, Float3 p2) {
	 return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z);
 }

 void Cpu_MC_Run(cube* dCubes, Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex) {
	 // Each index represents for a cube

	 for (int index = 0; index < param->cube_num; index++) {
#if (defined KERNEL)||(defined DF)
		 for (int i = 0; i < 8; i++) {
			 dCubes[index].vertices[i].val = 0;
			 Float3 vert = dCubes[index].vertices[i].pos;
			 if (vert.x == 0.0f || vert.x == param->worldSize.x || vert.y == 0.0f || vert.y == param->worldSize.y || vert.z == 0.0f || vert.z == param->worldSize.z)
				 continue;
			 Uint3 cellPos = Cpu_computeCellPosition(vert, param);
			 Uint3 pos;
			 for (int x = -1; x <= 1; x++)
				 for (int y = -1; y <= 1; y++)
					 for (int z = -1; z <= 1; z++) {
						 pos.x = cellPos.x + x;
						 pos.y = cellPos.y + y;
						 pos.z = cellPos.z + z;
						 int hash = Cpu_computeCellHash(pos, param);
						 if (hash < 0 || hash >= param->cells_total)
							 continue;
						 if (dStart[hash] < 0 || dStart[hash] >= param->num_particles)
							 continue;
						 uint count = 0;
						 for (count = dStart[hash]; count <= dEnd[hash]; count++) {
							 Particle *current = &particles[dParticleIndex[count]];
							 float distanceSquare = Cpu_dCalDistance((*current).pos, vert);
							 if (distanceSquare > param->h_square)
								 continue;
							 dCubes[index].vertices[i].val += param->mass * param->poly6_coff * pow(param->h_square - distanceSquare, 3) / current->dens;
						 }
					 }
		 }
#endif

#ifdef SPLINE_KERNEL
		 for (int i = 0; i < 8; i++) {
			 dCubes[index].vertices[i].val = 0;
			 Float3 vert = dCubes[index].vertices[i].pos;
			 if (vert.x == 0.0f || vert.x == param->worldSize.x || vert.y == 0.0f || vert.y == param->worldSize.y || vert.z == 0.0f || vert.z == param->worldSize.z)
				 continue;
			 Uint3 cellPos = Cpu_computeCellPosition(vert, param);
			 Uint3 pos;
			 ITERATE_NEIGHBOR{
				 pos.x = cellPos.x + x;
			 pos.y = cellPos.y + y;
			 pos.z = cellPos.z + z;
			 int hash = Cpu_computeCellHash(pos, param);
			 if (hash < 0 || hash >= param->cells_total)
				 continue;
			 if (dStart[hash] < 0 || dStart[hash] >= param->num_particles)
				 continue;
			 uint count = 0;
			 for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				 Particle *current = &particles[dParticleIndex[count]];
				 float distanceSquare = Cpu_dCalDistance((*current).pos, vert);
				 //if (distanceSquare<ZERO || distanceSquare>param->h_square)
				 float distance = sqrt(distanceSquare);
				 float q = distance / param->h;
				 if (q > 2 || q < 0)
					 continue;
				 if (q <= 0.5)
					 dCubes[index].vertices[i].val += param->mass * (1 - 1.5*pow(q, 2) + 0.75*pow(q, 3)) * param->spline_coff / current->dens;
				 else if (q < 1)
					 dCubes[index].vertices[i].val += param->mass * (0.25 * pow(2 - q, 3)) *param->spline_coff / current->dens;
			 }
			 }
		 }
#endif
	 }
 }

 void Cpu_MC_ComputeNormal(cube* dCubes, Param* param) {
	 for (int index = 0; index < param->cube_num; index++) {
		 int prev, next;
		 for (int i = 0; i < 8; i++) {
			 prev = index - 1;
			 next = index + 1;
			 if (prev < 0 && next >= param->cube_num)
				 dCubes[index].vertices[i].norm.x = 0.0f;
			 else if (prev < 0)
				 dCubes[index].vertices[i].norm.x = (0.0f - dCubes[next].vertices[i].val) / param->cubeSize;
			 else if (next >= param->cube_num)
				 dCubes[index].vertices[i].norm.x = (dCubes[prev].vertices[i].val - 0.0f) / param->cubeSize;
			 else {
				 dCubes[index].vertices[i].norm.x = (dCubes[prev].vertices[i].val - dCubes[next].vertices[i].val) / param->cubeSize;
			 }


			 prev = index - param->cubeCount.x;
			 next = index + param->cubeCount.x;
			 if (prev < 0 && next >= param->cube_num)
				 dCubes[index].vertices[i].norm.y = 0.0f;
			 else if (prev < 0)
				 dCubes[index].vertices[i].norm.y = (0.0f - dCubes[next].vertices[i].val) / param->cubeSize;
			 else if (next >= param->cube_num)
				 dCubes[index].vertices[i].norm.y = (dCubes[prev].vertices[i].val - 0.0f) / param->cubeSize;
			 else
				 dCubes[index].vertices[i].norm.y = (dCubes[prev].vertices[i].val - dCubes[next].vertices[i].val) / param->cubeSize;

			 prev = index - param->cubeCount.x * param->cubeCount.y;
			 next = index + param->cubeCount.x * param->cubeCount.y;
			 if (prev < 0 && next >= param->cube_num)
				 dCubes[index].vertices[i].norm.z = 0.0f;
			 else if (prev < 0)
				 dCubes[index].vertices[i].norm.z = (0.0f - dCubes[next].vertices[i].val) / param->cubeSize;
			 else if (next >= param->cube_num)
				 dCubes[index].vertices[i].norm.z = (dCubes[prev].vertices[i].val - 0.0f) / param->cubeSize;
			 else
				 dCubes[index].vertices[i].norm.z = (dCubes[prev].vertices[i].val - dCubes[next].vertices[i].val) / param->cubeSize;

			 float norm = -sqrt(dCubes[index].vertices[i].norm.x * dCubes[index].vertices[i].norm.x + dCubes[index].vertices[i].norm.y *dCubes[index].vertices[i].norm.y + dCubes[index].vertices[i].norm.z * dCubes[index].vertices[i].norm.z);

			 if (norm == 0.0f) {
				 dCubes[index].vertices[i].norm.x = dCubes[index].vertices[i].norm.y = dCubes[index].vertices[i].norm.z = 0.0f;
			 }
			 else {
				 dCubes[index].vertices[i].norm.x /= norm;
				 dCubes[index].vertices[i].norm.y /= norm;
				 dCubes[index].vertices[i].norm.z /= norm;
			 }
		 }
	 }
 }

 void Cpu_MC_Run2(cube* dCubes, Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex, Float3* dTriangles, Float3* dNorms) {
	 // Each index represents for a cube
	 for (int index = 0; index < param->cube_num; index++) {
		 for (int i = 0; i < 15; i++) {
			 dTriangles[15 * index + i].x = -999.0f;
			 dTriangles[15 * index + i].y = -999.0f;
			 dTriangles[15 * index + i].z = -999.0f;
			 dNorms[15 * index + i].x = -999.0f;
			 dNorms[15 * index + i].y = -999.0f;
			 dNorms[15 * index + i].z = -999.0f;
		 }

		 // For a particular cube
		 cube *c = &dCubes[index];

		 // Determine its situation
		 int flag_index = 0;
		 if (c->vertices[0].val < param->isovalue) flag_index |= 1;
		 if (c->vertices[1].val < param->isovalue) flag_index |= 2;
		 if (c->vertices[2].val < param->isovalue) flag_index |= 4;
		 if (c->vertices[3].val < param->isovalue) flag_index |= 8;
		 if (c->vertices[4].val < param->isovalue) flag_index |= 16;
		 if (c->vertices[5].val < param->isovalue) flag_index |= 32;
		 if (c->vertices[6].val < param->isovalue) flag_index |= 64;
		 if (c->vertices[7].val < param->isovalue) flag_index |= 128;

		 int edge_flags = Cpu_device_cube_edge_flags[flag_index];
		 if (edge_flags == 0)
			 continue;

		 // Based on its situation, compute the edge intersection
		 Float3 device_edge_vertex[12];
		 Float3 device_edge_norm[12];

		 for (uint count = 0; count < 12; count++)
		 {
			 if (edge_flags & (1 << count))
			 {
				 float diff = (param->isovalue - c->vertices[Cpu_device_edge_conn[count][0]].val) / (c->vertices[Cpu_device_edge_conn[count][1]].val - c->vertices[Cpu_device_edge_conn[count][0]].val);

				 device_edge_vertex[count].x = c->vertices[Cpu_device_edge_conn[count][0]].pos.x + (c->vertices[Cpu_device_edge_conn[count][1]].pos.x - c->vertices[Cpu_device_edge_conn[count][0]].pos.x) * diff;
				 device_edge_vertex[count].y = c->vertices[Cpu_device_edge_conn[count][0]].pos.y + (c->vertices[Cpu_device_edge_conn[count][1]].pos.y - c->vertices[Cpu_device_edge_conn[count][0]].pos.y) * diff;
				 device_edge_vertex[count].z = c->vertices[Cpu_device_edge_conn[count][0]].pos.z + (c->vertices[Cpu_device_edge_conn[count][1]].pos.z - c->vertices[Cpu_device_edge_conn[count][0]].pos.z) * diff;

				 device_edge_norm[count].x = c->vertices[Cpu_device_edge_conn[count][0]].norm.x + (c->vertices[Cpu_device_edge_conn[count][1]].norm.x - c->vertices[Cpu_device_edge_conn[count][0]].norm.x) * diff;
				 device_edge_norm[count].y = c->vertices[Cpu_device_edge_conn[count][0]].norm.y + (c->vertices[Cpu_device_edge_conn[count][1]].norm.y - c->vertices[Cpu_device_edge_conn[count][0]].norm.y) * diff;
				 device_edge_norm[count].z = c->vertices[Cpu_device_edge_conn[count][0]].norm.z + (c->vertices[Cpu_device_edge_conn[count][1]].norm.z - c->vertices[Cpu_device_edge_conn[count][0]].norm.z) * diff;
			 }
		 }

		 for (uint count_triangle = 0; count_triangle < 5; count_triangle++)
		 {
			 if (Cpu_device_triangle_table[flag_index][3 * count_triangle] < 0)
			 {
				 break;
			 }
			 for (uint count_point = 0; count_point < 3; count_point++)
			 {
				 int tt = Cpu_device_triangle_table[flag_index][3 * count_triangle + count_point];
				 dTriangles[15 * index + 3 * count_triangle + count_point].x = device_edge_vertex[tt].x;
				 dTriangles[15 * index + 3 * count_triangle + count_point].y = device_edge_vertex[tt].y;
				 dTriangles[15 * index + 3 * count_triangle + count_point].z = device_edge_vertex[tt].z;

				 dNorms[15 * index + 3 * count_triangle + count_point].x = device_edge_norm[tt].x;
				 dNorms[15 * index + 3 * count_triangle + count_point].y = device_edge_norm[tt].y;
				 dNorms[15 * index + 3 * count_triangle + count_point].z = device_edge_norm[tt].z;
			 }
		 }
	 }
 }

void Cpu_MC_RUN_ONE_TIME(cube *dCubes, Particle *dParticles, Param *param, uint* dStart, uint* dEnd, uint* dParticleIndex, Float3*dTriangles, Float3* dNorms, Param* hParam) {
	Cpu_MC_Run (dCubes, dParticles, param, dStart, dEnd, dParticleIndex);
	Cpu_MC_ComputeNormal(dCubes, param);
	Cpu_MC_Run2(dCubes, dParticles, param, dStart, dEnd, dParticleIndex, dTriangles, dNorms);
}


#endif