#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include "Wav.h"
#include "Math36.h"

const int kNumPrograms = 1;
const double soundSpeed = 3430.0; // cm per sec
// Min freq = 10Hz
// Max sampleRate = 96000Hz
const int shapeBuffSize = 9600;
const int sizePlot = 1024;
const int displayLoop = 50;

enum EParams
{
  kPower = 0,
  kType,
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

  void OnIdle() override;
  void OnUIClose() override;
  void OnUIOpen() override;

  double getFromX(const double& x);
  double getFromTheta(const double& x);
  double getSigmoid(const double& x);

  void createTopoCartesianShape();
  void createTopoPolarShape();
  void createSigmoidShape();

#endif
private:
  double power;
  Math36::Sigmoid sigmoid;

  bool UIClosed, isInit;
  int displayCount, atStartCount;
};
