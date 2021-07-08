#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"

const int kNumPrograms = 1;
const double soundSpeed = 3430.0; // cm per sec
const int maxBuffSize = 16384;

enum EParams
{
  kSpeakDelay = 0,
  kEarsDist,
  kSpeakAtt,
  kLtresh,
  kNumParams
};

using namespace iplug;
using namespace igraphics;

class Limiter {
public:
  enum LType {
    MIRROR = 0,
    TRUNC
  };
  Limiter();
  ~Limiter();

  void ProcessBlock(sample** toProcess, const sample& thresh, const int& nFrames, const int& from, const int& buffSize);
  void ProcessBlock(sample toProcess[2][maxBuffSize], const sample& thresh, const int& nFrames, const int& from, const int& buffSize);
  sample processSample(sample& smp, const sample& thresh);

  void setChannels(const int& n);
  sample algo(const sample& input, const sample& thresh);
  void setType(const LType& ltype);
  void setPeakSender(IPeakSender<2>* psender, const int& ctrlTag);

private:
  LType type;
  int nChans, cTag;
  sample** diff;
  IPeakSender<2>* sender;
};

class IPlugEffect final : public Plugin
{
public:
  IPlugEffect(const InstanceInfo& info);

#if IPLUG_DSP // http://bit.ly/2S64BDd
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
#endif
private:
  sample last_out_buffer[2][maxBuffSize];
  Limiter limiter;
};
