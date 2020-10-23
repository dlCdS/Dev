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
  kShift = 0,
  kMode,
  kLowShift,
  kHighShift,
  kPhase,
  kR,
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
  void OnReset() override;

  void TransformFFT(sample** inputs, sample** outputs, int nFrames);

  void GrohFreqShift(sample** inputs, sample** outputs, int nFrames);
  void SingleSSB(sample** inputs, sample** outputs, int nFrames);
  void AllpassSSB(sample** inputs, sample** outputs, int nFrames);
 
#endif
private:

  void setSoundTouch();

  void testPlug();

  sample buffer[2][maxBuffSize],
    sigCont[2];

  soundtouch::SAMPLETYPE  buffer_l[maxBuffSize],
    buffer_r[maxBuffSize];
  double phaseCont[maxBuffSize];
  CArray carray_l,
    carray_r;
  PArray parray_l,
    parray_r;

  sample filter1[2], filter2[2], freq_prev_arg[2],
    last_in1[2], last_in2[2];

  double sigVar[2], last_input[2];

  double shift, lastShift;
  int type;
  double lowShift;
  double highShift;
  double phase;
  double R;
  bool shiftChanged;

  Math36::AllpassFilter allpf0, allpf1;
  Math36::QSO qso1, qso2;

  soundtouch::SoundTouch _soundtouch[2];
  bool _stconfigured;
};
