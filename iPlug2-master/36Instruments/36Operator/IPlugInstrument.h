#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include <fstream>
#include "SineShape.h"

const int kNumPresets = 1;


enum EParams
{
  kParamGain = 0,

  kParamNoteGlideTime,
  kParamDoGlide,

  kParamVoices,

  kParamShape,
  kParamPhase,
  kParamPitch,
  kParamDetune,
  kParamSpread,

  kParamAttack,
  kParamDecay,
  kParamSustain,
  kParamRelease,

  kParamPitchBend,

  kPitchAttack,
  kPitchAttackValue,
  kPitchAttackTime,

  kLFOFilterType,

  kFilterAttack,
  kFilterAttackFromF,
  kFilterAttackFromN,
  kFilterAttackToF,
  kFilterAttackToN,
  kFilterAttackType,
  kFilterAttackAtt,
  kFilterAttackFollow,
  kFilterAttackTrig,
  kFilterAttackResFrom,
  kFilterAttackResTo,
  kFilterAttackTime,

  kParamLFOFilterShape,
  kParamLFOFilterRateHz,
  kParamLFOFilterRateTempo,
  kParamLFOFilterRateMode,
  kParamLFOFilterPhase,
  kParamLFOFilter,

  kParamLFOGainShape,
  kParamLFOGainRateHz,
  kParamLFOGainRateTempo,
  kParamLFOGainRateMode,
  kParamLFOGainPhase,
  kParamLFOGain,
  kNumParams
};

#if IPLUG_DSP
// will use EParams in IPlugInstrument_DSP.h
#include "IPlugInstrument_DSP.h"
#endif

enum EControlTags
{
  kCtrlTagMeter = 0,
  kCtrlTagLFOFilterVis,
  kCtrlTagLFOGainVis,
  kCtrlTagLFOPitchVis,
  kCtrlTagScope,
  kCtrlTagRTText,
  kCtrlTagKeyboard,
  kCtrlTagBender,
  kNumCtrlTags
};

using namespace iplug;
using namespace igraphics;

class IPlugInstrument final : public Plugin
{
public:
  IPlugInstrument(const InstanceInfo& info);

#if IPLUG_DSP // http://bit.ly/2S64BDd
public:
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
  void ProcessMidiMsg(const IMidiMsg& msg) override;
  void OnReset() override;
  void OnParamChange(int paramIdx) override;
  void OnIdle() override;
  bool OnMessage(int msgTag, int ctrlTag, int dataSize, const void* pData) override;

  void OnUIClose() override;
  void OnUIOpen() override;

private:
  SineShape sineshape;
  IPeakSender<2> mMeterSender;
  ISender<1> mLFOFilterVisSender, mLFOGainVisSender, mLFOPitchVisSender;

  bool UIClosed;
#endif
};