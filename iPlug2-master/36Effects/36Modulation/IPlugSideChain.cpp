#include "IPlugSideChain.h"
#include "IPlug_include_in_plug_src.h"

IPlugSideChain::IPlugSideChain(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets))
{
  GetParam(kModulType)->InitEnum("Type", 0, 5, "", IParam::kFlagsNone, "", "Mult", "Env", "Divide", "Warp", "Switch");
  GetParam(kIsSidechained)->InitEnum("Sidechained", 0, 2, "", IParam::kFlagsNone, "", "Nop", "Yes");

  GetParam(kDivide)->InitDouble("Divide", 1.0, 1.0, 10., 0.01);
  GetParam(kDivideFollow)->InitEnum("Normalize", 0, 2, "", IParam::kFlagsNone, "", "Nop", "Yes");

  GetParam(kSmooth)->InitEnum("Style", 0, 4, "", IParam::kFlagsNone, "", "Hard", "Soft", "Clic", "Always");

  

  GetParam(kDelay)->InitDouble("Delay", 0.0, 0.0, 50., 0.01, "ms");
  GetParam(kLookahead)->InitEnum("Lookahead", 0, 2, "", IParam::kFlagsNone, "", "Nop", "Yes");
  GetParam(kZeroTrunc)->InitEnum("ZeroTrunc", 0, 2, "", IParam::kFlagsNone, "", "Nop", "Yes");
  
  
  GetParam(kAbsolute)->InitEnum("Absolute", 0, 2, "", IParam::kFlagsNone, "", "Nop", "Yes");

  for (int i = 0; i < maxBuffSize; i++)
    for (int c = 0; c < 2; c++)
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
    const int nCols = 4;

    int cellIdx = -1;

    auto nextCell = [&]() {
      return b.GetGridCell(++cellIdx, nRows, nCols).GetPadded(-5.);
    };

    auto cell = [&](int r, int c) {
      return b.GetGridCell(r * nCols + c, nRows, nCols).GetPadded(-5.);
    };

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(0, 0).GetMidVPadded(buttonSize), kIsSidechained, "Sidechained", style, true), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 0).GetMidVPadded(buttonSize), kAbsolute, "Absolute", style, true),
      kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() > 3));

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 1).Union(cell(1, 3)).GetMidVPadded(buttonSize), 
      kSmooth, "Style", style, true), kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() < 4));

    pGraphics->AttachControl(new IVKnobControl(cell(1, 1).GetMidVPadded(buttonSize),
      kDivide, "Divide", style, false), kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() != 2));

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 2).GetMidVPadded(buttonSize),
      kDivideFollow, "Normalize", style, false), kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() != 2));


    pGraphics->AttachControl(new IVKnobControl(cell(1, 1).GetMidVPadded(buttonSize),
      kDelay, "Delay", style, false), kNoTag, "vcontrols")->Hide((GetParam(kModulType)->Value() != 3));

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 2).GetMidVPadded(buttonSize),
      kLookahead, "Lookahead", style, false), kNoTag, "vcontrols")->SetAnimationEndActionFunction([pGraphics](IControl* pControl) {
        bool sync = (pControl->GetValue() > 0.5);
        pGraphics->HideControl(kZeroTrunc, sync);
        })->Hide((GetParam(kModulType)->Value() != 3));

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 3).GetMidVPadded(buttonSize),
      kZeroTrunc, "Zero Trunc", style, false), kNoTag,
      "vcontrols")->Hide((GetParam(kModulType)->Value() != 3));
    

    pGraphics->AttachControl(new IVSlideSwitchControl(cell(0, 1).Union(cell(0, 3)).GetMidVPadded(buttonSize), kModulType,
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
  static int selectedChan[2] = { 0, 0 };
  static double der[4] = { 0.0, 0.0, 0.0, 0.0 };
  static double last[4] = { 0.0, 0.0, 0.0, 0.0 };
  const double epsilon = 0.0000001;
  const int smooth = GetParam(kSmooth)->Value();
  bool chanChange(false);
  const bool absolute = GetParam(kAbsolute)->Value();
  const bool follow = GetParam(kDivideFollow)->Value();
  const bool zeroTrunc = GetParam(kZeroTrunc)->Value();

  const int maxDelay = double(0.05 * GetSampleRate());
  const bool lookahead = GetParam(kLookahead)->Value();
  const double delay = GetParam(kDelay)->Value();
  static double lastDelay = 0.0; //used to smooth delay changes
  double curDelay = 0.0;
  double sampleDelay = delay / 1000. * GetSampleRate();
  double sampleOffset = 0.0; // Absolute position with warped modulation
  static int delayRef = 0; // Ref of start point for copying new buffer
  int curPos = 0, dprev, dnext; // Used for current, prev and next pos in buffer
  double factprev, factnext;

  if (sidechained && type == 3) { // copy buffer for warp algo
    for (int s = 0; s < nFrames; s++) 
      for (int c = 0; c < 2; c++) {
        delay_buffer[c][(s + delayRef) % maxBuffSize] = inputs[c][s];
      }
    if (lookahead)
      SetLatency(maxDelay);
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

        switch (type) {
        case 0:
          outputs[c][s] = inputs[c][s] * value;
          break;
        case 1:
          if(absolute)
            outputs[c][s] = inputs[c][s] * value;
          else outputs[c][s] = inputs[c][s] * (value + 1.0) / 2.0;
          break;
        case 2:
          if (absolute) value += divOffset;
          else value = (value + 1.0 ) / 2.0 + divOffset;
          if (value < divOffset) value = divOffset;

          outputs[c][s] = inputs[c][s] / (value * divide);
          break;
        case 3:
          // Warp
          curDelay = lastDelay + (delay - lastDelay) * ((double)s / (double)nFrames);
          curDelay = curDelay / 1000.0 * GetSampleRate();
          if (lookahead) curPos = (delayRef + s - maxDelay + maxBuffSize) % maxBuffSize;
          else curPos = (delayRef + s) % maxBuffSize;
          
          if(lookahead) value = inputs[c + 2][s];
          else if (zeroTrunc) {
            if (inputs[c + 2][s] > 0.0) {
              if (absolute) value = -1.0 * inputs[c + 2][s];
              else value = 0.0;
            }
            else value = inputs[c + 2][s];
          }
          else {
            value = inputs[c + 2][s] - 1.0;
          }
          
          if (delay < epsilon) {
            outputs[c][s] = inputs[c][curPos];
          }
          else {
            sampleOffset = (double)curPos + curDelay * value;
            if (sampleOffset < 0) sampleOffset += maxBuffSize;
            else if (sampleOffset >= maxBuffSize) sampleOffset -= maxBuffSize;

            dprev = ((int)sampleOffset);
            dnext = ((int)(sampleOffset + 1.0));
            factprev = sampleOffset - (double)dprev;
            factnext = (double)dnext - sampleOffset;
            dprev %= maxBuffSize;
            dnext %= maxBuffSize;
            outputs[c][s] = factprev * delay_buffer[c][dprev] + factnext * delay_buffer[c][dnext];
          }

          // Changing delay value while playing may cause errors, hack to avoid so
          // if (outputs[c][s] * outputs[c][s] >= 1.0) outputs[c][s] = 0.0;
          break;
        case 4:
          // Follow selected signal unless it is crossed by the other signal
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
