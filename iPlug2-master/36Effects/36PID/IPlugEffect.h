#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include "..\36Common\soundtouch\include\SoundTouch.h"

#include "..\36Common\Math36.h"


const int kNumPrograms = 1;
const double soundSpeed = 3430.0; // cm per sec
const int maxBuffSize = 16384;

enum EParams
{
  kP = 0,
  kI,
  kD,
  kMode,
  kNumParams
};

using namespace iplug;
using namespace igraphics;

class IPlugEffect final : public Plugin
{
public:
  IPlugEffect(const InstanceInfo& info);

#if IPLUG_DSP // http://bit.ly/2S64BDd
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
 
#endif
private:


  void testPlug();
  Math36::PID _pid[2];

  double last_out[2];
  
};
