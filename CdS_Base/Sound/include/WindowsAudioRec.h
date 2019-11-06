#pragma once
#include <Windows.h>
#include <mmsystem.h>
#include <fstream>
#include <iostream>
#include "SoundOperation.h"

class WindowsAudioRec
{
public:
	WindowsAudioRec();
	~WindowsAudioRec();

	void recordLoop();
};

