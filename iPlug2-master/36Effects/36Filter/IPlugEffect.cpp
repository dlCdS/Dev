#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include <iostream>
#include <fstream>


IPlugEffect::IPlugEffect(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPrograms))
{
  GetParam(kReal)->InitDouble("Real", 0.5, -1, 1, 0.001);
  GetParam(kImag)->InitDouble("Imag", 0.0, -1, 1, 0.001);





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

    const IVStyle style1{
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

    const IVStyle style2{
     false, // Show label
     true, // Show value
     {
       IColor(0, 255, 255, 255), // Background
       IColor(255, 255, 255, 255), // Foreground
       IColor(255, 255, 255, 255), // Pressed
       COLOR_BLACK, // Frame
       IColor(0, 255, 255, 255), // Highlight
       IColor(0, 255, 255, 255), // Shadow
       IColor(0, 255, 255, 255), // Extra 1
       IColor(0, 255, 255, 255), // Extra 2
       IColor(0, 255, 255, 255)  // Extra 3
     }, // Colors
     IText(20.f, EAlign::Center) // Label text
    };

    const IText forkAwesomeText{ 20.f, "ForkAwesome" };

    const int nRows = 5;
    const int nCols = 6;

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
      return b.GetGridCell(r*nCols+c, nRows, nCols).GetPadded(-5.);
    };
    
    auto sliderCell = [&](int r, int c, int size) {
      IRECT rec = cell(r, c).Union(cell(r + size, c));
      //IRECT rec = b.GetFromBottom(200.).GetFromLeft(200.+c*200.);
      rec.Translate(50.-c*11.0, -17.);
      rec.Scale(1.08);
      return rec;
    };

    /*
    const int cwidth = PLUG_WIDTH / nCols;
    const int cheigh = PLUG_HEIGHT / nRows;

    auto sliderCell = [&](int r, int c, int size) {
      return IRECT((r + 1) * cwidth, c * cheigh, cwidth, size * cheigh);
    };
    */
    {
    

      // upper left
      pGraphics->AttachControl(new IVKnobControl(cell(0, 3).GetMidVPadded(buttonSize), kReal, "real", style, false), kNoTag, "vcontrols");
      
      //down left
      pGraphics->AttachControl(new IVKnobControl(cell(1, 3).GetMidVPadded(buttonSize), kImag, "imag", style, false), kNoTag, "vcontrols");
     
       }

    // Left
   
  };

#endif
}

#if IPLUG_DSP


void IPlugEffect::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  const int nChans = NOutChansConnected();
  const double sampleRate = GetSampleRate();

  double real = GetParam(kReal)->Value();
  double imag = GetParam(kImag)->Value();

  for (int c = 0; c < nChans; c++)
    filter[c].setPole(Complex(real, imag));

  for (int c = 0; c < nChans; c++) {
    for (int i = 0; i < nFrames; i++) {
      outputs[c][i] = filter[c].process(inputs[c][i]);
    }
  }
 
}

#endif
