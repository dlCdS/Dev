#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"

const int kNumPrograms = 1;
const double soundSpeed = 3430.0; // cm per sec
const int maxBuffSize = 16384;

enum EParams
{
  kPan = 0,
  kEarsDist,
  kRLDist,
  kPronon,
  kMonoComp,
  kMonoLR,
  kMsType,
  kMsCenterSide,
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
  sample last_buffer[2][maxBuffSize],
    ms_m[maxBuffSize], ms_s[2][maxBuffSize], last_ms_s[2][maxBuffSize];
  bool work_buf_on; 
};
