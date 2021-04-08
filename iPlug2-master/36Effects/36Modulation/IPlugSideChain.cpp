#include "IPlugSideChain.h"
#include "IPlug_include_in_plug_src.h"

IPlugSideChain::IPlugSideChain(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets)), 
lastDelay(0.0),
delayRef(0),
noAmount(false),
hasLatency(false)
{
  memset(selectedChan, 0, sizeof(selectedChan));
  memset(der, 0.0, sizeof(der));
  memset(last, 0.0, sizeof(last));
  memset(lastoutput, 0.0, sizeof(lastoutput));
  memset(sideDb, 0.0, sizeof(sideDb));
  memset(inDerAvg, 0.0, sizeof(inDerAvg));

  GetParam(kModulType)->InitEnum("Type", 0, 5, "", IParam::kFlagsNone, "", "Mult", "Env", "Divide", "Warp", "Switch");
  GetParam(kIsSidechained)->InitEnum("Sidechained", 0, 2, "", IParam::kFlagsNone, "", "Nop", "Yes");

  GetParam(kAmount)->InitDouble("Amount", 1.0, .0, 1., 0.01);
  GetParam(kFromDB)->InitDouble("FromDB ", 0., 0., 1., 0.001);

  GetParam(kDivide)->InitDouble("Divide", 1.0, 1.0, 10., 0.01);
  GetParam(kDivideFollow)->InitEnum("Normalize", 0, 2, "", IParam::kFlagsNone, "", "Nop", "Yes");

  GetParam(kSmooth)->InitEnum("Style", 0, 4, "", IParam::kFlagsNone, "", "Hard", "Soft", "Clic", "Always");

  

  GetParam(kDelay)->InitDouble("Delay", 0.0, 0.0, 50., 0.01, "ms");
  GetParam(kLookahead)->InitEnum("Lookahead", 0, 2, "", IParam::kFlagsNone, "", "Nop", "Yes");
  GetParam(kZeroTrunc)->InitEnum("ZeroTrunc", 0, 2, "", IParam::kFlagsNone, "", "Nop", "Yes");
  
  
  GetParam(kAbsolute)->InitEnum("Absolute", 0, 2, "", IParam::kFlagsNone, "", "Nop", "Yes");

  for (int i = 0; i < maxBuffSize; i++)
    for (int c = 0; c < 4; c++)
      delay_buffer[c][i] = 0.0;

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

    const int nRows = 2;
    const int nCols = 5;

    int cellIdx = -1;

    auto nextCell = [&]() {
      return b.GetGridCell(++cellIdx, nRows, nCols).GetPadded(-5.);
    };

    auto cell = [&](int r, int c) {
      return b.GetGridCell(r * nCols + c, nRows, nCols).GetPadded(-5.);
    };

    pGraphics->AttachControl(new IVKnobControl(cell(0, 0).GetMidVPadded(buttonSize),
      kAmount, "Amount", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(cell(0, 1).GetMidVPadded(buttonSize),
      kFromDB, "From DB", style, false), kNoTag, "vcontrols");

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 0).GetMidVPadded(buttonSize), kIsSidechained, "Sidechained", style, true), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 1).GetMidVPadded(buttonSize), kAbsolute, "Absolute", style, true),
      kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() > 3));

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 2).Union(cell(1, 4)).GetMidVPadded(buttonSize), 
      kSmooth, "Style", style, true), kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() < 4));

    pGraphics->AttachControl(new IVKnobControl(cell(1, 2).GetMidVPadded(buttonSize),
      kDivide, "Divide", style, false), kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() != 2));

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 3).GetMidVPadded(buttonSize),
      kDivideFollow, "Normalize", style, false), kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() != 2));


    pGraphics->AttachControl(new IVKnobControl(cell(1, 2).GetMidVPadded(buttonSize),
      kDelay, "Delay", style, false), kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() != 3));

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 3).GetMidVPadded(buttonSize),
      kLookahead, "Lookahead", style, false), kNoTag, "vcontrols")->SetAnimationEndActionFunction([pGraphics](IControl* pControl) {
        bool sync = (pControl->GetValue() > 0.5);
        pGraphics->HideControl(kZeroTrunc, sync);
        })->Hide((GetParam(kModulType)->Value() != 3));

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 4).GetMidVPadded(buttonSize),
      kZeroTrunc, "Zero Trunc", style, false), kNoTag,
      "vcontrols")->Hide((GetParam(kModulType)->Value() != 3));
    

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(0, 2).Union(cell(0, 4)).GetMidVPadded(buttonSize), kModulType,
      "Modulation Type", style, true), kNoTag, "vcontrols")->SetAnimationEndActionFunction([pGraphics](IControl* pControl) {
      bool sync = (pControl->GetValue() <= 0.4 || pControl->GetValue() > 0.6);
      bool sync2 = (pControl->GetValue() < 0.9);
      bool sync3 = (pControl->GetValue() <= 0.6 || pControl->GetValue() > 0.8);
      bool sync4 = (pControl->GetValue() > 0.8);
      pGraphics->HideControl(kDivide, sync);
      pGraphics->HideControl(kDivideFollow, sync);
      pGraphics->HideControl(kSmooth, sync2);
      pGraphics->HideControl(kDelay, sync3);
      pGraphics->HideControl(kLookahead, sync3);
      pGraphics->HideControl(kZeroTrunc, sync3);
      pGraphics->HideControl(kAbsolute, sync4);
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

  const double epsilon = 0.0000001;
  const int smooth = GetParam(kSmooth)->Value();
  bool chanChange(false);
  const bool absolute = GetParam(kAbsolute)->Value();
  const bool follow = GetParam(kDivideFollow)->Value();
  const bool zeroTrunc = GetParam(kZeroTrunc)->Value();

  const int maxDelay = 0.05 * GetSampleRate();
  const bool lookahead = GetParam(kLookahead)->Value();
  const double delay = GetParam(kDelay)->Value();
  double curDelay = 0.0;
  double sampleDelay = delay / 1000. * GetSampleRate();
  double sampleOffset = 0.0; // Absolute position with warped modulation
  int curPos = 0, dprev, dnext; // Used for current, prev and next pos in buffer
  double factprev, factnext;

  const double baseAmount = GetParam(kAmount)->Value();
  double amount = baseAmount;
  const double fromDb = GetParam(kFromDB)->Value();

  const double derCoef = 0.9;

  if (sidechained && type == 3) { // copy buffer for warp algo
    for (int s = 0; s < nFrames; s++) 
      for (int c = 0; c < nChans; c++) 
        delay_buffer[c][(s + delayRef) % maxBuffSize] = inputs[c][s];

    if (lookahead) {
      if (!hasLatency) {
        SetLatency(maxDelay);
        hasLatency = true;
      }
    }
    else if (hasLatency) {
        SetLatency(0);
        hasLatency = false;
    }
  } else if (hasLatency) {
    SetLatency(0);
    hasLatency = false;
  }

  double divOffset = 1.0;
  if (follow)
    divOffset = 1.0 / divide;
  
  double value;

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
        if (absolute)
          value = abs(inputs[c + 2][s]);
        else value = inputs[c + 2][s];

        sideDb[c] = (1.0 - fromDb) * sideDb[c] + abs(inputs[c + 2][s]);

        if (sideDb[c] >= 1.0) sideDb[c] = 1.0;
        else if (sideDb[c] < epsilon) sideDb[c] = 0.0;

        amount = sideDb[c] * baseAmount;

        switch (type) {
        case 0:
          // neutral = 1.0
          value = amount * value + (1.0 - amount);

          outputs[c][s] = inputs[c][s] * value;
          break;

        case 1:
          // neutral = 1.0
          value = amount * value + (1.0 - amount);

          if(absolute)
            outputs[c][s] = inputs[c][s] * value;
          else outputs[c][s] = inputs[c][s] * (value + 1.0) / 2.0;
          break;

        case 2:
          
          if (absolute) value += divOffset;
          else value = (value + 1.0 ) / 2.0 + divOffset;
          if (value < divOffset) value = divOffset;

          // neutral = 1.0 / divide
          value = amount * value + (1.0 - amount) / divide;

          outputs[c][s] = inputs[c][s] / (value * divide);
          break;

        case 3:
          // Warp
          curDelay = lastDelay + (delay - lastDelay) * ((double)s / (double)nFrames);
          curDelay = curDelay / 1000.0 * GetSampleRate(); // amount controls

          if (lookahead) curPos = (delayRef + s - maxDelay + maxBuffSize) % maxBuffSize;
          else curPos = (delayRef + s) % maxBuffSize;

          if (!lookahead) {
            if (absolute) value *= - 1.0;
            else value -= 1.0;
          }
          
           sampleOffset = double(curPos + maxBuffSize) + curDelay * value * amount;

           dprev = ((int)sampleOffset);
           dnext = dprev + 1;
           factnext = sampleOffset - (double)dprev;
           factprev = 1.0 - factnext;
           dprev %= maxBuffSize;
           dnext %= maxBuffSize;
           outputs[c][s] = factprev * delay_buffer[c][dprev] + factnext * delay_buffer[c][dnext];

          break;

        case 4:
          // Follow selected signal unless it is crossed by the other signal
          // amount not applicable
          if ((inputs[c][s] - inputs[c + 2][s]) * (inputs[c][s] - inputs[c + 2][s]) < epsilon) { // signal crossing
            if (smooth == 3) selectedChan[c] = (selectedChan[c] + 1) % 2;
            else {
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
              }
              else if (smooth == 2) chanChange = true;

              if (chanChange) { // same variation direction
                if (inputs[c][s] >= 0) {
                  if (inputs[c][s] > inputs[c + 2][s]) selectedChan[c] = 0;
                  else selectedChan[c] = 1;
                }
                else {
                  if (inputs[c][s] < inputs[c + 2][s]) selectedChan[c] = 0;
                  else selectedChan[c] = 1;
                }
              }
            }
          }
          if(selectedChan[c] == 0)
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
  for (int c = 0; c < 2; c++) {
    lastoutput[c] = outputs[c][nFrames - 1];
  }

  delayRef = (delayRef + nFrames) % maxBuffSize;
  lastDelay = delay;
  
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
