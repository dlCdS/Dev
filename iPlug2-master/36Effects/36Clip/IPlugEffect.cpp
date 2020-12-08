#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include <iostream>
#include <fstream>

#include "IconsForkAwesome.h"

const int shuff = 3;
const double highCap = 3.0;
const double vhighCap = 9.0;

double displayTest(const double& x, IPlugEffect* plug) {
  return plug->dampTest[int(x * (displaySize - shuff))];
}

double displayDamp(const double& x, IPlugEffect* plug) {
  return plug->dampRes[int(x * (displaySize - shuff))];
}

double displayCapped(const double& x, IPlugEffect* plug) {
  return plug->dampCap[int(x * (displaySize - shuff))]/plug->sder_cap;
}

double displayHigh(const double& x, IPlugEffect* plug) {
  return plug->dampHigh[int(x * (displaySize - shuff))]/ highCap;
}

double displayVHigh(const double& x, IPlugEffect* plug) {
  return plug->dampVHigh[int(x * (displaySize - shuff))] / vhighCap;
}

IPlugEffect::IPlugEffect(const InstanceInfo& info) 
: Plugin(info, MakeConfig(kNumParams, kNumPrograms)), der_disp(0.0), UIClosed(true), displayCount(0), isInit(false),
atStartCount(0), sampleRate(0), is_active(false)
{
  GetParam(kCap)->InitDouble("Cap", 1., 0., 1., 0.001, "");
  GetParam(kDamp)->InitDouble("Damp", 1., 1., 3., 0.001, "");
  GetParam(kSmooth)->InitDouble("Smooth", 0., 0., 1., 0.001, "");
  GetParam(kTresh)->InitDouble("Tresh", 1., 0., .5, 0.001, "");
  GetParam(kWorkGain)->InitDouble("WorkGain", 1., 0.1, 20., 0.001, "");
  GetParam(kDisplay)->InitBool("Display", true);
  GetParam(kLog)->InitBool("Log", true);

  for (int s = 0; s < maxBuffSize; s++) {
    for (int i = 0; i < 2; i++) {
      der_sec[i][s] = 0.0;
      correct[i][s] = 0.0;
    }
  }

  for (int i = 0; i < 2; i++) {
    prev_sig[i][0] = 0.0;
    prev_sig[i][1] = 0.0;
    prev_orig[i] = 0.0;
  }

  sder_cap = 1.0;

  for (int s = 0; s < displaySize; s++) {
    if (s < displaySize / 2.0)
      dampTest[s] = 1.0;
    else
      dampTest[s] = 0.0;

      dampRes[s] = 0.0;
      dampCap[s] = 0.0;
      dampHigh[s] = 0.0;
      dampVHigh[s] = 0.0;
  }
  dampTest[0] = 0.0;

  computeDisplay();


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

    const IVStyle style_green{
     true, // Show label
     true, // Show value
     {
       IColor(0, 255, 200, 100), // Background
       IColor(200, 200, 200, 230), // Foreground
       DEFAULT_PRCOLOR, // Pressed
       COLOR_BLACK, // Frame
       DEFAULT_HLCOLOR, // Highlight
       DEFAULT_SHCOLOR, // Shadow
       COLOR_GREEN, // Extra 1
       DEFAULT_X2COLOR, // Extra 2
       DEFAULT_X3COLOR  // Extra 3
     }, // Colors
     IText(16.f, EAlign::Center) // Label text
    };

    const IVStyle style_blue{
     true, // Show label
     true, // Show value
     {
       IColor(0, 255, 200, 100), // Background
       IColor(200, 200, 200, 230), // Foreground
       DEFAULT_PRCOLOR, // Pressed
       COLOR_BLACK, // Frame
       DEFAULT_HLCOLOR, // Highlight
       DEFAULT_SHCOLOR, // Shadow
       COLOR_BLUE, // Extra 1
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

    const IVStyle style_red{
     true, // Show label
     true, // Show value
     {
       IColor(0, 255, 200, 100), // Background
       IColor(200, 200, 200, 230), // Foreground
       DEFAULT_PRCOLOR, // Pressed
       COLOR_BLACK, // Frame
       DEFAULT_HLCOLOR, // Highlight
       DEFAULT_SHCOLOR, // Shadow
       COLOR_RED, // Extra 1
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

    const IText forkAwesomeText{ 20.f, "ForkAwesome" };

    const int nRows = 6;
    const int nCols = 5;

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

    
    pGraphics->AttachControl(new IVKnobControl(cell(0, 0).GetMidVPadded(buttonSize), kCap, "Cap", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(cell(0, 1).GetMidVPadded(buttonSize), kDamp, "Damp", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(cell(0, 2).GetMidVPadded(buttonSize), kSmooth, "Smooth", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(cell(0, 3).GetMidVPadded(buttonSize), kTresh, "Tresh", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(cell(0, 4).GetMidVPadded(buttonSize), kWorkGain, "Work Gain", style, false), kNoTag, "vcontrols");



    pGraphics->AttachControl(new IVPlotControl(cell(3, 0).Union(cell(5, 4)), {
      {COLOR_DARK_GRAY , [&](double x) { return displayTest(x, this); } },
      {COLOR_BLUE , [&](double x) { return displayDamp(x, this); } },
      {COLOR_GREEN , [&](double x) { return displayCapped(x, this); } },
      {COLOR_VIOLET , [&](double x) { return displayHigh(x, this); } },
      {COLOR_RED , [&](double x) { return displayVHigh(x, this); } }
      
      }, displaySize-1, "", style, -0.2, 1.2), kNoTag, "vcontrols");

    
    pGraphics->AttachControl(new IVDisplayControl(cell(1, 0).Union(cell(2, 4)), "", style_green, EDirection::Horizontal, 0., 1., 0., sizePlot), kNumParams, "LFO");
    pGraphics->AttachControl(new IVDisplayControl(cell(1, 0).Union(cell(2, 4)), "", style_blue, EDirection::Horizontal, 0., 1., 0., sizePlot), kNumParams+1, "LFO");
    pGraphics->AttachControl(new IVDisplayControl(cell(1, 0).Union(cell(2, 4)), "", style_violet, EDirection::Horizontal, 0., 1., 0., sizePlot), kNumParams+2, "LFO");
    pGraphics->AttachControl(new IVDisplayControl(cell(1, 0).Union(cell(2, 4)), "", style_red, EDirection::Horizontal, 0., 1., 0., sizePlot), kNumParams+3, "LFO");
    pGraphics->AttachControl(new IVDisplayControl(cell(1, 0).Union(cell(2, 4)), "", style_gray, EDirection::Horizontal, 0., 1., 0., sizePlot), kNumParams+4, "LFO");

    const double dist = 30.;
    pGraphics->AttachControl(new ITextToggleControl(cell(1, 0).GetGridCell(0, 0, 2, 2).GetFromTop(dist).GetFromLeft(dist), kDisplay, ICON_FK_SQUARE_O, ICON_FK_CHECK_SQUARE, forkAwesomeText), kNoTag, "vcontrols");
    pGraphics->AttachControl(new ITextToggleControl(cell(1, 0).GetGridCell(0, 1, 2, 2).GetFromTop(dist).GetFromLeft(dist), kLog, ICON_FK_SQUARE_O, ICON_FK_CHECK_SQUARE, forkAwesomeText), kNoTag, "vcontrols");
  };



#endif
  //testPlug();
}



#if IPLUG_DSP

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

bool checkChanged(double& var, const double& src) {
  if (var == src)
    return false;
  var = src;
  return true;
}

void IPlugEffect::ProcessBlock(sample** inputs, sample** outputs, int nFrames) {
  const int nChans = NOutChansConnected();
  bool changed(false);


  if (sampleRate != GetSampleRate()) {
    sampleRate = GetSampleRate();
    global_ts = 48000 / sampleRate;
  }

  changed = changed || checkChanged(sder_cap, GetParam(kCap)->Value());
  changed = changed || checkChanged(damp, GetParam(kDamp)->Value());
  changed = changed || checkChanged(smooth, GetParam(kSmooth)->Value());

  smooth = GetParam(kSmooth)->Value()* GetParam(kSmooth)->Value();

  tresh = GetParam(kTresh)->Value();
  double wgain = GetParam(kWorkGain)->Value();
  wgain *= wgain;

  bool use_log10 = GetParam(kLog)->Value(),
    print_disp = GetParam(kDisplay)->Value();

  

  if (!UIClosed){
    if (atStartCount < 100) {
      atStartCount++;
      changed = true;
    }
    if (changed && atStartCount >= 100) {
      computeDisplay();
      GetUI()->SetAllControlsDirty();
    }
  }
 
    for (int s = 0; s < nFrames; s++) {
      der_disp = 0.0;
      for (int chan = 0; chan < nChans; chan++) {
        outputs[chan][s] = correction(inputs[chan][s] * wgain, prev_sig[chan][0], prev_sig[chan][1], prev_orig[chan], global_ts, &der_disp);
        prev_sig[chan][1] = prev_sig[chan][0];
        prev_sig[chan][0] = outputs[chan][s];
        prev_orig[chan] = inputs[chan][s] * wgain;
        outputs[chan][s] /= wgain;
      }

      if(use_log10 && print_disp){
        mRLSender.PushData({ kNumParams, {float(log10(tresh * sder_cap + 1.0))} });
        mRLSender.PushData({ kNumParams + 1, {float(log10(tresh + 1.0))} });
        mRLSender.PushData({ kNumParams + 2, {float(log10(tresh * highCap + 1.0))} });
        mRLSender.PushData({ kNumParams + 3, {float(log10(tresh * vhighCap + 1.0))} });
        mRLSender.PushData({ kNumParams + 4, {float(log10(der_disp + 1.0))} });
      }
      else if(print_disp) {
        mRLSender.PushData({ kNumParams, {float((tresh * sder_cap ))} });
        mRLSender.PushData({ kNumParams + 1, {float((tresh ))} });
        mRLSender.PushData({ kNumParams + 2, {float((tresh * highCap ))} });
        mRLSender.PushData({ kNumParams + 3, {float((tresh * vhighCap ))} });
        mRLSender.PushData({ kNumParams + 4, {float((der_disp ))} });
      }
  }



  isInit = true;
  ++displayCount %= displayLoop;
}

double IPlugEffect::correction(const double& sig, const double& sig1, const double& sig2, const double& base_sig1, const double& ts, double* display_der)
{
  double der2 = (sig + sig2 - 2.0 * sig1) / ts / ts,
    der_init = (sig - base_sig1) / ts,
    der, loc_sder_cap(sder_cap / ts / ts);
  double sig_res = sig;
  bool do_damp = false;

  if (display_der != NULL) {
    if (abs(der2 * ts * ts) > * display_der)
      *display_der = abs(der2 * ts * ts);
    loc_sder_cap *= tresh;
  }

  if (abs(der2) > loc_sder_cap) {
    if (der2 < 0) der2 = -1.0 * loc_sder_cap;
    else der2 = loc_sder_cap;
    sig_res = (der2 * ts * ts + 2.0 * sig1 - sig2) / damp;
  }


  der = (sig_res - sig1) / ts;
  if(der != der_init) {
    der = smooth * der_init + (1.0 - smooth) * der;
    sig_res = der * ts + sig1;
  }
   

  return sig_res;
}

void IPlugEffect::computeDisplay()
{
  for (int i = 2; i < displaySize; i++) {
    dampRes[i] = correction(dampTest[i], dampRes[i - 1], dampRes[i - 2], dampTest[i-1]);
    dampCap[i] = correction(dampTest[i]*sder_cap, dampCap[i - 1], dampCap[i - 2], dampTest[i - 1] * sder_cap);
    dampHigh[i] = correction(dampTest[i]* highCap, dampHigh[i - 1], dampHigh[i - 2], dampTest[i - 1] * highCap);
    dampVHigh[i] = correction(dampTest[i] * vhighCap, dampVHigh[i - 1], dampVHigh[i - 2], dampTest[i - 1] * vhighCap);
  }
}

const std::string file_path = "E:\\\Programmes\\VS2017\\iPlug2-master\\36Effects\\36PID\\build-win\\test.txt";


#endif
