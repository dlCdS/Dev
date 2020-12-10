#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include <iostream>

double displayShape(const double& x, IPlugEffect* plug) {
  return plug->getFromX(x);
}

double displayShapeNeg(const double& x, IPlugEffect* plug) {
  return plug->getFromX(x + 1.0);
}

double displayShapePolar(const double& x, IPlugEffect* plug) {
  return plug->getFromTheta(x);
}

IPlugEffect::IPlugEffect(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPrograms)), power(1.0), UIClosed(true), displayCount(0), isInit(false),
atStartCount(0)
{
  GetParam(kPower)->InitDouble("Pan", 2., .1, 10.0, 0.01);
  GetParam(kType)->InitEnum("Type", 0, 3, "", IParam::kFlagsNone, "", "None", "Side", "M/S");

#if IPLUG_EDITOR // http://bit.ly/2S64BDd
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

    const int nRows = 8;
    const int nCols = 8;

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

    pGraphics->AttachControl(new IVKnobControl(cell(0, 0).GetMidVPadded(buttonSize), kPower, "Pan", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVSlideSwitchControl(cell(0, 1).GetMidVPadded(buttonSize), kType, "M/S type", style, true), kNoTag, "vcontrols");


    pGraphics->AttachControl(new IVPlotControl(cell(1, 0).Union(cell(3, 2)), {
     {COLOR_DARK_GRAY , [&](double x) { return displayShape(x, this); } },
     {COLOR_DARK_GRAY , [&](double x) { return displayShapeNeg(x, this); } },
      }, sizePlot, "", style, -0.85, .85), kNoTag, "vcontrols");

    pGraphics->AttachControl(new IVPlotControl(cell(1, 3).Union(cell(3, 7)), {
     {COLOR_DARK_GRAY , [&](double x) { return displayShape(x*2.0, this); } },
      }, sizePlot, "", style, -0.85, .85), kNoTag, "vcontrols");
      
    pGraphics->AttachControl(new IVPlotControl(cell(4, 3).Union(cell(6, 7)), {
     {COLOR_DARK_GRAY , [&](double x) { return displayShapePolar(x * 2.0, this); } },
      }, sizePlot, "", style, -0.85, .85), kNoTag, "vcontrols");
  };
#endif
}

#if IPLUG_DSP

void IPlugEffect::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  const int nChans = NOutChansConnected();
  const double sampleRate = GetSampleRate();
  bool changed = (power == GetParam(kPower)->Value());
  power = GetParam(kPower)->Value();

  if (!UIClosed) {
    if (atStartCount < 100) {
      atStartCount++;
      changed = true;
    }
    if (changed && atStartCount >= 100) {
      GetUI()->SetAllControlsDirty();
    }
  }

  
  int MS_type = GetParam(kType)->Value();


  isInit = true;
  ++displayCount %= displayLoop;
}

void IPlugEffect::OnIdle()
{
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

double IPlugEffect::getFromX(const double& val)
{
  if (val >= 1.0) return -1.0 * getFromX(val - 1.0);
  else if(val<0.0) return -1.0 * getFromX(val + 1.0);
  double x = val * 2.0 - 1.0;
  return pow(1.0 - pow(abs(x), power), 1.0 / power);
}
double IPlugEffect::getFromTheta(const double& val)
{
  double x = val * PI;
  return pow(1.0 / (pow(abs(cos(x)), power) + pow(abs(sin(x)), power)), 1.0 / power) * sin(x);
}
#endif
