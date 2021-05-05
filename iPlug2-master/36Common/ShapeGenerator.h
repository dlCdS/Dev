#pragma once
#include "IPlugInstrument.h"
#include <mutex>

class ShapeGenerator
{
public:
  ShapeGenerator();

  void ProcessMidiMsg(const IMidiMsg &msg);
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames, int sampleRate);

private:
  bool play;
  std::mutex mutex;
  bool freq, last_freq;
};

