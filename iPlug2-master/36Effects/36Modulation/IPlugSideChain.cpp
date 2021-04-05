#include "IPlugSideChain.h"
#include "IPlug_include_in_plug_src.h"

IPlugSideChain::IPlugSideChain(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets))
{
  GetParam(kModulType)->InitEnum("Type", 0, 4, "", IParam::kFlagsNone, "", "Mult", "Env", "Divide", "Switch");
  GetParam(kIsSidechained)->InitEnum("Sidechained", 0, 2, "", IParam::kFlagsNone, "", "Nop", "Yes");

  GetParam(kDivide)->InitDouble("Divide", 1.0, 1.0, 10., 0.01);
  GetParam(kSmooth)->InitEnum("Style", 0, 3, "", IParam::kFlagsNone, "", "H", "S", "C");


  mMakeGraphicsFunc = [&]() {
    return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS, 1.);
  };
  
  mLayoutFunc = [&](IGraphics* pGraphics) {
    pGraphics->AttachCornerResizer(EUIResizerMode::Scale, false);
    pGraphics->AttachPanelBackground(IColor(255, 255, 190, 100));
    pGraphics->LoadFont("Roboto-Regular", ROBOTO_FN);

    pGraphics->AttachBubbleControl();

    const IRECT b = pGraphics->GetBounds().GetPadded(-5);

    const float buttonSize = 45.f;

    const IVStyle style{
     true, // Show label
     true, // Show value
     {
       IColor(255, 255, 200, 100), // Background
       IColor(255, 255, 240, 230), // Foreground
       DEFAULT_PRCOLOR, // Pressed
       COLOR_BLACK, // Frame
       DEFAULT_HLCOLOR, // Highlight
       DEFAULT_SHCOLOR, // Shadow
       COLOR_BLACK, // Extra 1
       DEFAULT_X2COLOR, // Extra 2
       DEFAULT_X3COLOR  // Extra 3
     }, // Colors
     IText(16.f, EAlign::Center) // Label text
    };

    const IText forkAwesomeText{ 20.f, "ForkAwesome" };

    const int nRows = 1;
    const int nCols = 4;

    int cellIdx = -1;

    auto nextCell = [&]() {
      return b.GetGridCell(++cellIdx, nRows, nCols).GetPadded(-5.);
    };

    auto cell = [&](int r, int c) {
      return b.GetGridCell(r * nCols + c, nRows, nCols).GetPadded(-5.);
    };

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(0, 0).GetMidVPadded(buttonSize), kIsSidechained, "Sidechained", style, true), kNoTag, "vcontrols");

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(0, 3).GetMidVPadded(buttonSize), 
      kSmooth, "Style", style, true), kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() <= 0.7));

    pGraphics->AttachControl(new IVKnobControl(cell(0, 3).GetMidVPadded(buttonSize),
      kDivide, "Divide", style, false), kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() < 0.5 || GetParam(kModulType)->Value() > 0.7));

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(0, 1).Union(cell(0, 2)).GetMidVPadded(buttonSize), kModulType, "Modulation Type", style, true), kNoTag, "vcontrols")->SetAnimationEndActionFunction([pGraphics](IControl* pControl) {
      bool sync = (pControl->GetValue() < 0.5 || pControl->GetValue() > 0.7);
      bool sync2 = (pControl->GetValue() <= 0.7);
      pGraphics->HideControl(kDivide, sync);
      pGraphics->HideControl(kSmooth, sync2);
      });

  };

}

#if IPLUG_DSP

void IPlugSideChain::GetBusName(ERoute direction, int busIdx, int nBuses, WDL_String& str) const
{
  //could customize bus names here
  IPlugProcessor::GetBusName(direction, busIdx, nBuses, str);
}

void IPlugSideChain::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  const int nChans = NOutChansConnected();
  const int type = GetParam(kModulType)->Value();
  const int sidechained = GetParam(kIsSidechained)->Value();
  const double divide = GetParam(kDivide)->Value();
  static int selectedChan = 0;
  static double der[4] = { 0.0, 0.0, 0.0, 0.0 };
  static double last[4] = { 0.0, 0.0, 0.0, 0.0 };
  const double epsilon = 0.0000001;
  const int smooth = GetParam(kSmooth)->Value();
  bool chanChange(false);

  for (int i=0; i < 4; i++) {
    bool connected = IsChannelConnected(ERoute::kInput, i);
    if(connected != mInputChansConnected[i]) {
      mInputChansConnected[i] = connected;
      mSendUpdate = true;
    }
  }
  
  for (int i=0; i < 2; i++) {
    bool connected = IsChannelConnected(ERoute::kOutput, i);
    if(connected != mOutputChansConnected[i]) {
      mOutputChansConnected[i] = connected;
      mSendUpdate = true;
    }
  }
  
  for (int s = 0; s < nFrames; s++) {
    for (int c = 0; c < 2; c++) {
      if (mInputChansConnected[c + 2] && sidechained) {
        switch (type) {
        case 0:
          outputs[c][s] = inputs[c][s] * inputs[c + 2][s];
          break;
        case 1:
            outputs[c][s] = inputs[c][s] * (inputs[c + 2][s] + 1.0) / 2.0;
          break;
        case 2:
          if ((inputs[c + 2][s] + 2.0) >= 1.0)
          outputs[c][s] = inputs[c][s] / ((inputs[c + 2][s] + 2.0)/2.0 * divide);
          else outputs[c][s] = inputs[c][s];
          break;
        case 3:
          // Follow selected signal unless it is crossed by the other signal
          if ((inputs[c][s] - inputs[c + 2][s]) * (inputs[c][s] - inputs[c + 2][s]) < epsilon) { // signal crossing
            if (s == 0) {
              der[c] = inputs[c][s] - last[c];
              der[c + 2] = inputs[c + 2][s] - last[c + 2];
            }
            else {
              der[c] = inputs[c][s] - inputs[c][s - 1];
              der[c + 2] = inputs[c + 2][s] - inputs[c + 2][s - 1];
            }
            chanChange = false;
            if (smooth == 0)
              chanChange = true;
            else if (der[c] * der[c + 2] >= 0) {
              if (smooth == 1) chanChange = true;
            } else if (smooth == 2) chanChange = true;

            if (chanChange) { // same variation direction
              if (inputs[c][s] >= 0) {
                if (inputs[c][s] > inputs[c + 2][s]) selectedChan = 0;
                else selectedChan = 1;
              }
              else {
                if (inputs[c][s] < inputs[c + 2][s]) selectedChan = 0;
                else selectedChan = 1;
              }
            }
          }
          if(selectedChan == 0)
            outputs[c][s] = inputs[c][s];
          else
            outputs[c][s] = inputs[c + 2][s];
          break;
        default:
          outputs[c][s] = inputs[c][s];
          break;
        }
      }
      else outputs[c][s] = inputs[c][s];
    }
  }

  for (int c = 0; c < nChans; c++) {
    last[c] = inputs[c][nFrames - 1];
  }
  
  /*
    
     Logic/Garageband have an long-standing bug where if no sidechain is selected, the same buffers that are sent to the first bus, are sent to the sidechain bus
     https://forum.juce.com/t/sidechain-is-not-silent-as-expected-au-logic-x-10-2-2/17068/8
     https://lists.apple.com/archives/coreaudio-api/2012/Feb/msg00127.html
   
     Imperfect hack around it here. Probably a better solution is to have an enable sidechain button in the plug-in UI, in addition to the host sidechain routing.
  */
  
#if defined OS_MAC && defined AU_API
  if(GetHost() == kHostLogic || GetHost() == kHostGarageBand) {
    const int sz = nFrames * sizeof(sample);
    if(!memcmp(inputs[0], inputs[2], sz)) {
      memset(inputs[2], 0, sz);
      mInputChansConnected[2] = false;
    }
    if(!memcmp(inputs[1], inputs[3], sz)) {
      memset(inputs[3], 0, sz);
      mInputChansConnected[3] = false;
    }
  }
#endif

}
#endif
