// #pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include "config.h"
#include "../36Common/Math36.h"
// #include "Common.h"

const int kNumPrograms = 1;
const double soundSpeed = 3430.0; // cm per sec
const int maxBuffSize = 16384; // 800;
// const int maxBuffSize = 2 * PLUG_LATENCY;

const bool testPlugin = false;

enum EParams
{
  kReal = 0,
  kImag,
  kAtt,
  type,
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

  Math36::Filter filter[2];
};
