#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include "..\36Common\soundtouch\include\SoundTouch.h"

#include "..\36Common\Math36.h"


const int kNumPrograms = 1;
const double soundSpeed = 3430.0; // cm per sec
const int maxBuffSize = 16384;
const int displaySize = 60;
const int displayLoop = 50;
const int sizePlot = 1024;
const int derDispSize = 10;

enum EParams
{
  kDamp = 0,
  kCap,
  kSmooth,
  kTresh,
  kWorkGain,
  kDisplay,
  kLog,
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
 
#endif
private:
    double der_sec[2][maxBuffSize],
    correct[2][maxBuffSize];

    ISender<1> mRLSender;

    bool UIClosed, isInit;
    int displayCount, atStartCount;

public:
  double dampTest[displaySize],
    dampRes[displaySize],
    dampCap[displaySize],
    dampHigh[displaySize],
    dampVHigh[displaySize],
    prev_sig[2][2], prev_orig[2][2],
    global_ts, sampleRate,
    auto_scale;

  bool is_active;

  double sder_cap,
    smooth, tresh, smooth2;
  double derDisp[derDispSize], derDispInit[derDispSize];

  int dispcount;

  double correction(const double& sig, const double& sig1, const double& sig2,
    const double& base_sig1, const double& base_sig2,
    const double& ts = 1.0, double *display_der=NULL, double* display_der_init = NULL);

  void computeDisplay();
};
