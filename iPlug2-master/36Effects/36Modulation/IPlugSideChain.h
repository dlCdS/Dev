#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "ISender.h"
#include "IControls.h"

const int kNumPresets = 1;
const int maxBuffSize = 14000;

enum EParams
{
  kModulType = 0,
  kInputGain,
  kAmount,
  kFromDB,
  kIsSidechained,
  kDivide,
  kDivideFollow,
  kSmooth,
  kAbsolute,
  kDelay,
  kLookahead,
  kNumParams
};

enum ECtrlTags
{
  kCtrlTagMeter = 0,
  kNumCtrlTags
};

enum EMsgTags
{
  kMsgTagConnectionsChanged = 0,
  kNumMsgTags
};

using namespace iplug;
using namespace igraphics;

class IPlugSideChain final : public Plugin
{
public:
  IPlugSideChain(const InstanceInfo& info);

  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
  
  void GetBusName(ERoute direction, int busIdx, int nBuses, WDL_String& str) const override;
  void OnActivate(bool enable) override;

  void OnIdle() override;

private:
  bool mInputChansConnected[4] = {};
  bool mOutputChansConnected[2] = {};
  bool mSendUpdate = false;

  sample delay_buffer[4][maxBuffSize];
  sample input_buffer[2][2048];

  int selectedChan[2];
  double der[4];
  double last[4];
  double lastoutput[2];
  double lastDelay; //used to smooth delay changes
  int delayRef; // Ref of start point for copying new buffer
  double sideDb[2]; // measure of sidechain volume
  bool noAmount;
  double inDerAvg[4]; // derivative avg of input
  bool hasLatency;

  IPeakSender<4> mInputPeakSender;
  IVMeterControl<4>* mInputMeter = nullptr;
};
