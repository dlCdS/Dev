#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include <iostream>
#include <fstream>
#include <math.h>

#include "IconsForkAwesome.h"

double displayTest(const double& x, IPlugEffect* plug) {
  return plug->sigmoid.get(x * 10. - 5. + 0.5)-0.5;
}


IPlugEffect::IPlugEffect(const InstanceInfo& info) 
: Plugin(info, MakeConfig(kNumParams, kNumPrograms)), UIClosed(true), displayCount(0), isInit(false),
atStartCount(0), dispcount(0)
{
  GetParam(kTresh)->InitDouble("Tresh", 1., 0.01, 2., 0.01, "");
  GetParam(kStiff)->InitDouble("Stiff", 1., 0., 10.0, 0.01, "");

  GetParam(kGain)->InitDouble("Gain", 1., 0.1, 3., 0.001, "");
  GetParam(kInGain)->InitDouble("Input Gain", 1., 0.1, 20., 0.001, "");
  GetParam(kDisplay)->InitBool("Display", true);
  GetParam(kLog)->InitBool("Log", true);

  for (int i = 0; i < derDispSize; i++) {
    input_env[i] = 0.0;
    output_env[i] = 0.0;
  }


#if IPLUG_EDITOR // http://bit.ly/2S64BDd
  mMakeGraphicsFunc = [&]() {
    return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS, 1.);
  };
  
  mLayoutFunc = [&](IGraphics* pGraphics) {
    pGraphics->AttachCornerResizer(EUIResizerMode::Scale, false);
    pGraphics->AttachPanelBackground(IColor(255, 255, 190, 100));
    pGraphics->LoadFont("Roboto-Regular", ROBOTO_FN);
    pGraphics->LoadFont("ForkAwesome", FORK_AWESOME_FN);

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

    const IVStyle style_gray{
     true, // Show label
     true, // Show value
     {
       IColor(0, 255, 200, 100), // Background
       IColor(200, 200, 200, 230), // Foreground
       DEFAULT_PRCOLOR, // Pressed
       COLOR_BLACK, // Frame
       DEFAULT_HLCOLOR, // Highlight
       DEFAULT_SHCOLOR, // Shadow
       COLOR_DARK_GRAY, // Extra 1
       DEFAULT_X2COLOR, // Extra 2
       DEFAULT_X3COLOR  // Extra 3
     }, // Colors
     IText(16.f, EAlign::Center) // Label text
    };

    const IVStyle style_white{
     true, // Show label
     true, // Show value
     {
       IColor(0, 255, 200, 100), // Background
       IColor(200, 200, 200, 230), // Foreground
       DEFAULT_PRCOLOR, // Pressed
       COLOR_BLACK, // Frame
       DEFAULT_HLCOLOR, // Highlight
       DEFAULT_SHCOLOR, // Shadow
       COLOR_WHITE, // Extra 1
       DEFAULT_X2COLOR, // Extra 2
       DEFAULT_X3COLOR  // Extra 3
     }, // Colors
     IText(16.f, EAlign::Center) // Label text
    };

    const IVStyle style_violet{
     true, // Show label
     true, // Show value
     {
       IColor(0, 255, 200, 100), // Background
       IColor(200, 200, 200, 230), // Foreground
       DEFAULT_PRCOLOR, // Pressed
       COLOR_BLACK, // Frame
       DEFAULT_HLCOLOR, // Highlight
       DEFAULT_SHCOLOR, // Shadow
       COLOR_VIOLET, // Extra 1
       DEFAULT_X2COLOR, // Extra 2
       DEFAULT_X3COLOR  // Extra 3
     }, // Colors
     IText(16.f, EAlign::Center) // Label text
    };

    const IText forkAwesomeText{ 20.f, "ForkAwesome" };

    const int nRows = 3;
    const int nCols = 4;

    int cellIdx = -1;

    auto nextCell = [&]() {
      return b.GetGridCell(++cellIdx, nRows, nCols).GetPadded(-5.);
    };

    auto sameCell = [&]() {
      return b.GetGridCell(cellIdx, nRows, nCols).GetPadded(-5.);
    };
     
    auto AddLabel = [&](const char* label) {
      pGraphics->AttachControl(new ITextControl(nextCell().GetFromTop(20.f), label, style.labelText));
    };

    auto cell = [&](int r, int c) {
      return b.GetGridCell(r * nCols + c, nRows, nCols).GetPadded(-5.);
    };


    pGraphics->AttachControl(new IVKnobControl(cell(0, 0).GetMidVPadded(buttonSize), kInGain, "Input Gain", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(cell(0, 1).GetMidVPadded(buttonSize), kTresh, "Tresh", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(cell(0, 2).GetMidVPadded(buttonSize), kStiff, "Stiff", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(cell(0, 3).GetMidVPadded(buttonSize), kGain, "Gain", style, false), kNoTag, "vcontrols");


    pGraphics->AttachControl(new IVDisplayControl(cell(1, 0).Union(cell(2, 3)), "", style_gray, EDirection::Horizontal, 0., 1., 0., sizePlot), kNumParams, "LFO");
    pGraphics->AttachControl(new IVDisplayControl(cell(1, 0).Union(cell(2, 3)), "", style_white, EDirection::Horizontal, 0., 1., 0., sizePlot), kNumParams + 1, "LFO");
    pGraphics->AttachControl(new IVDisplayControl(cell(1, 0).Union(cell(2, 3)), "", style_violet, EDirection::Horizontal, 0., 1., 0., sizePlot), kNumParams + 2, "LFO");

    const double dist = 30.;
    pGraphics->AttachControl(new ITextToggleControl(cell(1, 0).GetGridCell(0, 0, 2, 2).GetFromTop(dist).GetFromLeft(dist), kDisplay, ICON_FK_SQUARE_O, ICON_FK_CHECK_SQUARE, forkAwesomeText), kNoTag, "vcontrols");
    pGraphics->AttachControl(new ITextToggleControl(cell(1, 0).GetGridCell(0, 1, 2, 2).GetFromTop(dist).GetFromLeft(dist), kLog, ICON_FK_SQUARE_O, ICON_FK_CHECK_SQUARE, forkAwesomeText), kNoTag, "vcontrols");
    
  };

  

#endif
}


#if IPLUG_DSP

void IPlugEffect::ProcessBlock(sample** inputs, sample** outputs, int nFrames) {
  const int nChans = NOutChansConnected();
  const double sampleRate = GetSampleRate();
  double stiff = GetParam(kStiff)->Value();
  bool changed = sigmoid.setSteepness(stiff* stiff);
  
  double tresh = GetParam(kTresh)->Value();
  tresh *= tresh;

  double gain = GetParam(kGain)->Value();
  double inGain = GetParam(kInGain)->Value();
  gain *= gain;
  inGain *= inGain;

  bool use_log10 = GetParam(kLog)->Value(),
    print_disp = GetParam(kDisplay)->Value();

  double inenv, outenv;

  if (!UIClosed)
    if (atStartCount < 100) atStartCount++;
    else if (displayCount % 1 == 0) GetUI()->SetAllControlsDirty();

  for (int i = 0; i < nFrames; i++) {
    inenv = 0.0;
    outenv = 0.0;
    for (int j = 0; j < 2; j++){
      outputs[j][i] = gain * 2.0 * tresh * (sigmoid.get(inGain * inputs[j][i] / tresh / 2.0 + 0.5) - 0.5) / inGain;
      inenv = std::max(inenv, inGain * inputs[j][i]);
      outenv = std::max(outenv, outputs[j][i]);
    }

    if (print_disp && !UIClosed) {
      input_env[dispcount] = inenv;
      output_env[dispcount] = outenv;
      dispcount = (dispcount + 1) % derDispSize;
      inenv = 0.0;
      outenv = 0.0;
      for (int i = 0; i < derDispSize; i++) {
        inenv = std::max(inenv, input_env[i]);
        outenv = std::max(outenv, output_env[i]);
      }

      if (use_log10) {
        mRLSender.PushData({ kNumParams, {float(log(inenv  + 1.0))} });
        mRLSender.PushData({ kNumParams + 1, {float(log(tresh + 1.0))} });
        mRLSender.PushData({ kNumParams + 2, {float(log(outenv  + 1.0))} });
      }
      else {
        mRLSender.PushData({ kNumParams, {float((inenv))} });
        mRLSender.PushData({ kNumParams + 1, {float((tresh))} });
        mRLSender.PushData({ kNumParams + 2, {float((tresh * outenv))} });
      }
    }

  }




  isInit = true;
  ++displayCount %= displayLoop;

}

void IPlugEffect::OnIdle()
{
  mRLSender.TransmitData(*this);
}

void IPlugEffect::OnUIClose()
{
  Plugin::OnUIClose();
  GetUI()->SetAllControlsClean();
  UIClosed = true;
  Sleep(500);
}

void IPlugEffect::OnUIOpen()
{
  Plugin::OnUIOpen();
  displayCount = 0;
  UIClosed = false;
}

const std::string file_path = "E:\\\Programmes\\VS2017\\iPlug2-master\\36Effects\\36PID\\build-win\\test.txt";


#endif
