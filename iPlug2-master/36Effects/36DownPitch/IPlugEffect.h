#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include "..\36Common\soundtouch\include\SoundTouch.h"

#include "..\36Common\Math36.h"


const int kNumPrograms = 1;
const double soundSpeed = 3430.0; // cm per sec
const int maxBuffSize = 16384;
const int displayLoop = 50;
const int sizePlot = 1024;
const int derDispSize = 10;

enum EParams
{
  kPitch = 0,
  kMaxDelay,
  kNumParams
};

using namespace iplug;
using namespace igraphics;

class Pitcher {
public:
  Pitcher();
  ~Pitcher();

  void ProcessBlock(sample** inputs, sample** outputs, int nFrames, int channel);

private:
  sample buffer[maxBuffSize];
  int cur_pos, next_pos;
  double rate, delay;
};

class IPlugEffect final : public Plugin
{
public:
  IPlugEffect(const InstanceInfo& info);

#if IPLUG_DSP // http://bit.ly/2S64BDd
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;

  void OnIdle() override;
  void OnUIClose() override;
  void OnUIOpen() override;

#endif
private:
  bool UIClosed, isInit;
  int displayCount, atStartCount;


  ISender<1> mRLSender;

  double input_env[derDispSize], output_env[derDispSize];

  int dispcount;

public:
  Math36::Sigmoid sigmoid;
  
};
