#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include <fstream>
#include "projects/SineShape.h"

const int kNumPresets = 1;

const double primeList[] ={
2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151,
1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223,
1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289,
1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367,
1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447,
1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499,
1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579,
1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637,
1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723,
1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801,
1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879,
1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979,
1987, 1993, 1997, 1999
};

const int primeListSize = sizeof(primeList) / sizeof(double);


namespace {
  const int nSample = 9600;
  double sine[primeListSize][nSample];
  double triangle[primeListSize][nSample];
  double square[primeListSize][nSample];
  double sawup[primeListSize][nSample];
  double sawdown[primeListSize][nSample];

  inline void init() {
    // sine
    for (int i = 0; i < primeListSize; i++) {
      for (int j = 0; j < nSample; j++) {
        sine[i][j] = sin(primeList[i] * 2.0 * M_PI * double(j) / double(nSample));
      }
    }
  }
};

enum EParams
{
  kParamGain = 0,
  kParamNoteGlideTime,
  kParamDoGlide,
  kParamVoices,
  kParamAttack,
  kParamDecay,
  kParamSustain,
  kParamRelease,
  kParamLFOShape,
  kParamLFORateHz,
  kParamLFORateTempo,
  kParamLFORateMode,
  kParamLFODepth,
  kNumParams
};

#if IPLUG_DSP
// will use EParams in IPlugInstrument_DSP.h
#include "IPlugInstrument_DSP.h"
#endif

enum EControlTags
{
  kCtrlTagMeter = 0,
  kCtrlTagLFOVis,
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

private:
  SineShape sineshape;
  IPeakSender<2> mMeterSender;
  ISender<1> mLFOVisSender;

#endif
};
