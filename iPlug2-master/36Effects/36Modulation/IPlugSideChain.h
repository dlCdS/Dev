#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "ISender.h"
#include "IControls.h"

const int kNumPresets = 1;
const int maxBuffSize = 14000;

enum EParams
{
  kModulType = 0,
  kIsSidechained,
  kDivide,
  kDivideFollow,
  kSmooth,
  kAbsolute,
  kDelay,
  kLookahead,
  kZeroTrunc,
  kNumParams
};

enum ECtrlTags
{
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

  bool mInputChansConnected[4] = {};
  bool mOutputChansConnected[2] = {};
  bool mSendUpdate = false;

  sample delay_buffer[2][maxBuffSize];
  
  IPeakSender<4> mInputPeakSender;
  IPeakSender<2> mOutputPeakSender;
  IVMeterControl<4>* mInputMeter = nullptr;
  IVMeterControl<2>* mOutputMeter = nullptr;
};
