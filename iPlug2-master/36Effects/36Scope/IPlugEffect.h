// #pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include "config.h"
#include <unordered_map>
#include <mutex>
#include "LFO.h"
// #include "Common.h"

const int kNumPrograms = 1;
/// bartime = 60 / bpm [s]
/// barbuffer = bartime * samplerate
/// maxbuffer = maxbar [2] * 60 / min_bpm [50] * max_samplerate [98000]
/// maxbuffer = 235200
const int maxScopeBuffSize = 1000; // 800;
// const int maxBuffSize = 2 * PLUG_LATENCY;
const int displayLoop = 50;

const bool testPlugin = false;



enum EParams
{
  dGrid,
  dStart,
  dSize,
  dZoom,
  dChannel,
  kNumParams
};

enum EControlTags
{
  kCtrlTagLim = 0,
  kCtrlTagOutput,
  kNumCtrlTags
};

using namespace iplug;
using namespace igraphics;

const double TempoDivisonToDouble[] =
{
  0.0625,   // 1 sixty fourth of a beat
  0.125,       // 1 thirty second of a beat
  0.166666,      // 1 sixteenth note tripet
  0.25,       // 1 sixteenth note
  0.333333,      // 1 dotted sixteenth note
  0.666666,        // 1 eigth note      // Corrected mess arounds definition and plugin display
  0.5,       // 1 dotted eigth note       // Corrected mess arounds definition and plugin display
  0.625,       // 1 eigth note tripet        // Corrected mess arounds definition and plugin display
  1.0,        // 1 quater note a.k.a 1 beat @ 4/4
  1.5,       // 1 dotted beat @ 4/4
  2.0,        // 2 beats @ 4/4
  4.0,          // 1 bar @ 4/4
  8.0,          // 2 bars @ 4/4
  16.0,          // 4 bars @ 4/4
  32.0          // 8 bars @ 4/4
};


class IPlugEffect final : public Plugin
{
public:
  IPlugEffect(const InstanceInfo& info);

#if IPLUG_DSP // http://bit.ly/2S64BDd
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
  const void updateDisplay(double& var, EParams param, bool &update);
  void dcBlock(sample** inputs);
  void OnIdle() override;
  void OnUIClose() override;
  void OnUIOpen() override;
  sample getValue(const sample& s, const int& side);
#endif
private:

  void testPlug();
  void updateCurve();

  IPeakSender<2> mOutSender, mLimSender;


  int displayCount, atStartCount;

public:
  double start, size, old_zoom;

  bool UIClosed, isInit;
  bool printR, printL, printMono;
  sample buffer[2][maxScopeBuffSize],
    mono[maxScopeBuffSize],
    phantom[2][maxScopeBuffSize];
  LFO<sample> lfo;

};
