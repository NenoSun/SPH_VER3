#pragma once
#include "Configuration.h"
#ifdef ENABLE_FRAME_CAPTURE
#ifdef WINDOWS
#include <windows.h>
#include <string>
using namespace std;

class Snapshot
{
public:
	Snapshot();
	~Snapshot();

	static int count;
	static void gdiscreen();//POINT a, POINT b);
	static int GetEncoderClsid(const WCHAR* format, CLSID* pClsid);
};

#endif
#endif

