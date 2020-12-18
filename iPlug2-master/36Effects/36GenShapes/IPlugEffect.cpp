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
  GetParam(kPower)->InitDouble("Pan", 2., -1.0, 10.0, 0.01);
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
  /*
  createTopoCartesianShape();
  createTopoPolarShape();
  createSigmoidShape();
  */
#endif
}

#if IPLUG_DSP

void IPlugEffect::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  const int nChans = NOutChansConnected();
  const double sampleRate = GetSampleRate();
  bool changed = (power == GetParam(kPower)->Value());
  power = GetParam(kPower)->Value();

  if (!UIClosed || true) {
    if (atStartCount < 100) {
      atStartCount++;
      changed = true;
    }
    if ((changed && atStartCount >= 100) || true) {
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
  if (power > 0.0)
    return pow(1.0 - pow(abs(x), power), 1.0 / power);
  else return 1.0;
}
double IPlugEffect::getFromTheta(const double& val)
{
  double x = val * PI;
  if(power > 0.0)
    return pow(1.0 / (pow(abs(cos(x)), power) + pow(abs(sin(x)), power)), 1.0 / power) * sin(x);
  else {
    if (val >= 1.0) return -1.0 * getFromTheta(val - 1.0);
    else return 1.0 / max(abs(cos(x)), abs(sin(x))) * sin(x);
  }
}

double IPlugEffect::getSigmoid(const double& val)
{
  double x = val;
  while (x > 2.0) x -= 2.0;
  if (x > 1.0) return getSigmoid(2.0 - x);
  return 2.0 * sigmoid.get(x) - 1.0;
}

void IPlugEffect::createTopoCartesianShape()
{
  const int  nTests = 13;

  WavFile::WavFileData* data = WavFile::initFile("TopoCartesianShape.wav");
  power = 0.5;
  for (int s = 0; s < shapeBuffSize; s++) 
    WavFile::printSignal(data, getFromX(2.0 * double(s) / shapeBuffSize));

  power = 0.65;
 for (int s = 0; s < shapeBuffSize; s++)
    WavFile::printSignal(data, getFromX(2.0 * double(s) / shapeBuffSize));

 power = 0.85;
 for (int s = 0; s < shapeBuffSize; s++)
   WavFile::printSignal(data, getFromX(2.0 * double(s) / shapeBuffSize));

 power = 1.;
  for (int i = 0; i < nTests; i++) {
    for (int s = 0; s < shapeBuffSize; s++) {
      WavFile::printSignal(data, getFromX(2.0 * double(s) / shapeBuffSize));
    }
    power *= pow(2.0, 0.25);
  }

  power = -1.;
  for (int s = 0; s < shapeBuffSize; s++)
    WavFile::printSignal(data, getFromX(2.0 * double(s) / shapeBuffSize));
  WavFile::closeFile(data);
}

void IPlugEffect::createTopoPolarShape()
{
  const int  nTests = 13;

  WavFile::WavFileData* data = WavFile::initFile("TopoPolarShape.wav");
  power = 0.5;
  for (int s = 0; s < shapeBuffSize; s++)
    WavFile::printSignal(data, getFromTheta(2.0 * double(s) / shapeBuffSize));

  power = 0.65;
  for (int s = 0; s < shapeBuffSize; s++)
    WavFile::printSignal(data, getFromTheta(2.0 * double(s) / shapeBuffSize));

  power = 0.85;
  for (int s = 0; s < shapeBuffSize; s++)
    WavFile::printSignal(data, getFromTheta(2.0 * double(s) / shapeBuffSize));

  power = 1.;
  for (int i = 0; i < nTests; i++) {
    for (int s = 0; s < shapeBuffSize; s++) {
      WavFile::printSignal(data, getFromTheta(2.0 * double(s) / shapeBuffSize));
    }
    power *= pow(2.0, 0.25);
  }

  power = -1.;
  for (int s = 0; s < shapeBuffSize; s++)
    WavFile::printSignal(data, getFromTheta(2.0 * double(s) / shapeBuffSize));
  WavFile::closeFile(data);
}

void IPlugEffect::createSigmoidShape()
{
  const int  nTests = 13;
  double stiff = -32;
  WavFile::WavFileData* data = WavFile::initFile("SigmoidShape.wav");

  while (stiff < -0.24) {
    sigmoid.setSteepness(stiff);
    for (int s = 0; s < shapeBuffSize; s++)
      WavFile::printSignal(data, getSigmoid(2.0 * double(s) / shapeBuffSize +.5));
    stiff /= 2.0;
  }

  sigmoid.setSteepness(0.0);
  for (int s = 0; s < shapeBuffSize; s++)
    WavFile::printSignal(data, getSigmoid(2.0 * double(s) / shapeBuffSize + .5));


  sigmoid.setSteepness(0.25);
  for (int s = 0; s < shapeBuffSize; s++)
    WavFile::printSignal(data, getSigmoid(2.0 * double(s) / shapeBuffSize + .5));


  sigmoid.setSteepness(0.5);
  for (int s = 0; s < shapeBuffSize; s++)
    WavFile::printSignal(data, getSigmoid(2.0 * double(s) / shapeBuffSize + .5));

  stiff = 1.;
  while (stiff < 200) {
    sigmoid.setSteepness(stiff);
    for (int s = 0; s < shapeBuffSize; s++)
      WavFile::printSignal(data, getSigmoid(2.0 * double(s) / shapeBuffSize + .5));
    stiff *= 1.5;
  }

  WavFile::closeFile(data);
}
#endif
