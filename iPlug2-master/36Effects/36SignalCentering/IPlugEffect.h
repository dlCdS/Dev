#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"

const int kNumPrograms = 1;
#define nOffset  10

enum EParams
{
  kGlide = 0,
  kDisplayOnly,
  kNumParams
};

enum EControlTags {
  kOffsetL = 0,
  kOffsetR,
  kNumCtrlTags
};

using namespace iplug;
using namespace igraphics;

class IPlugEffect final : public Plugin
{
public:
  IPlugEffect(const InstanceInfo& info);
  void OnIdle() override;

#if IPLUG_DSP // http://bit.ly/2S64BDd
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
#endif
  double offset[nOffset];
  ISender<1> mRLSender;
};
