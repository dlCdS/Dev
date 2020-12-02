#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include <iostream>
#include <fstream>


IPlugEffect::IPlugEffect(const InstanceInfo& info) 
: Plugin(info, MakeConfig(kNumParams, kNumPrograms))
{
  GetParam(kP)->InitDouble("p", 0.3, -0.3, .99, 0.01, "");
  GetParam(kI)->InitDouble("i", .8, 0.1, 10.0, 0.01, "");
  GetParam(kD)->InitDouble("d", 1.0, -1.0, 2.0, 0.01, "");
  GetParam(kMode)->InitEnum("Type", 0, 3, "", IParam::kFlagsNone, "", "Weaver", "Single SSB", "FFT");


  for (int i = 0; i < 2; i++) {
    last_out[i] = 0.0;
  }


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

    const int nRows = 2;
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


    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kP, "p", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kI, "i", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kD, "d", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVSlideSwitchControl(nextCell().GetMidVPadded(buttonSize), kMode, "mode", style, true), kNoTag, "vcontrols");
  };
  

#endif
  //testPlug();
}


#if IPLUG_DSP

void IPlugEffect::ProcessBlock(sample** inputs, sample** outputs, int nFrames) {
  const int nChans = NOutChansConnected();
  const double sampleRate = GetSampleRate();
  Math36::setSampleRate(sampleRate);
  for (int i = 0; i < 2; i++)
    _pid[i].setPID(GetParam(kP)->Value(), GetParam(kI)->Value(), GetParam(kD)->Value());

  //_pid[0].setTimestep(1.0 / sampleRate);
  //_pid[1].setTimestep(1.0 / sampleRate);

  for (int i = 0; i < nFrames; i++) {
    for (int j = 0; j < 2; j++){
      //outputs[j][i] = last_out[j];
      last_out[j] = _pid[j].get(inputs[j][i], last_out[j]);

      if (_audiodb[j].get(last_out[j]) > 0.0)
        last_out[j] = 0.0;
      outputs[j][i] = last_out[j];
      
        
    }
  }
  
}


const std::string file_path = "E:\\\Programmes\\VS2017\\iPlug2-master\\36Effects\\36PID\\build-win\\test.txt";


void IPlugEffect::testPlug()
{
  std::fstream file(file_path, std::ios::out | std::ios::trunc);

  const int size = 500,
    nloop = 10,
    gate_s = 300,
    gate_e = 0;
  sample** inputs, ** outputs;

  outputs = new sample * [2];
  inputs = new sample * [2];

  for (int i = 0; i < 2; i++) {
    outputs[i] = new sample[size];
    inputs[i] = new sample[size];
  }

  for (int l = 0; l < nloop; l++) {
    for (int s = 0; s < size; s++) {
      for (int i = 0; i < 2; i++) {
        inputs[i][s] = 0.25 * sin(double(s + l * size) * 0.01)
          + 0.15 * sin(double(s + l * size) * 0.006)
          + 0.1 * sin(double(s + l * size) * 0.03) + 0.2
          + rand() % 10 * 0.002;

        if (s + l * size > gate_s && s + l * size < gate_e)
          inputs[i][s] = 0.0;
      }
    }

    ProcessBlock(inputs, outputs, size);

    for (int s = 0; s < size; s++) {
      file << s + l * size << '\t' << inputs[0][s] << '\t' << outputs[0][s] << std::endl;
    }
  }


  for (int i = 0; i < 2; i++) {
    delete[] outputs[i];
    delete[] inputs[i];
  }
  delete[] outputs;
  delete[] inputs;

  file.close();
  system("pause");
}

#endif
