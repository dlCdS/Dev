#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include <iostream>
#include <fstream>


IPlugEffect::IPlugEffect(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPrograms))
{
  GetParam(kShift)->InitDouble("Shift", 0.0, 0.0, 3000, 0.01, "hz");
  GetParam(kMode)->InitEnum("Type", 0, 2, "", IParam::kFlagsNone, "", "Weaver", "Single SSB");

  GetParam(kLowShift)->InitDouble("Low Shift", 0.0, 0.0, 25000.0, 30.0, "hz");
  GetParam(kHighShift)->InitDouble("High Shift", 25000.0, 0.0, 25000.0, 30.0);
  GetParam(kPhase)->InitDouble("Phase", 0.0, 0.0, 1.0, 0.01);
  GetParam(kR)->InitDouble("R", 0.998, 0.0, 1.0, 0.001);

  carray_l.resize(maxBuffSize);
  carray_r.resize(maxBuffSize);
  parray_l.resize(maxBuffSize);
  parray_r.resize(maxBuffSize);


  for (int i = 0; i < maxBuffSize; i++) {
    for (int j = 0; j < 2; j++) {
      buffer[j][i] = 0;
      sigCont[j] = 0.0;
      filter1[j] = 0.0;
      filter2[j] = 0.0;
      last_in1[j] = 0.0;
      last_in2[j] = 0.0;
      freq_prev_arg[j] = 0.0;
      sigVar[j] = 1.0;
      last_input[j] = 0.0;
     }
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


    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kShift, "shift", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVSlideSwitchControl(nextCell().GetMidVPadded(buttonSize), kMode, "mode", style, true), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kLowShift, "lowshift", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kR, "R", style, false), kNoTag, "vcontrols");
  };
  

#endif
  //testPlug();
}


#if IPLUG_DSP

void IPlugEffect::ProcessBlock(sample** inputs, sample** outputs, int nFrames) {
  const int nChans = NOutChansConnected();
  const double sampleRate = GetSampleRate();
  Math36::setSampleRate(sampleRate);

  shift = GetParam(kShift)->Value();
  if (lastShift != shift) {
    lastShift = shift;
    shiftChanged = true;
  }
  else shiftChanged = false;
  type = GetParam(kMode)->Value();
  lowShift = GetParam(kLowShift)->Value();
  highShift = GetParam(kHighShift)->Value();
  phase = GetParam(kPhase)->Value();
  R= GetParam(kR)->Value();

  switch (type) {
  case 0:
    GrohFreqShift(inputs, outputs, nFrames);
    break;
  case 1:
    SingleSSB(inputs, outputs, nFrames);
    break;
  default:
    break;
  }
}


void IPlugEffect::TransformFFT(sample** inputs, sample** outputs, int nFrames)
{
  const int nChans = NOutChansConnected();
  const double sampleRate = GetSampleRate();

  double cur_freq;

  for (int frame = 0; frame < nFrames; frame++) {
    carray_l[frame] = Complex((double)inputs[0][frame], 0.0);
    carray_r[frame] = Complex((double)inputs[1][frame], 0.0);
  }

  // FFT

  Math36::fft(carray_l, nFrames);
  Math36::fft(carray_r, nFrames);

  //hilbert(carray_l, nFrames);
  //hilbert(carray_r, nFrames);

  // Convert in polar
  for (int x = 0; x < nFrames / 2; x++) {

    carray_l[x].imag(carray_l[x].imag() *shift);
    carray_r[x].imag(carray_r[x].imag() * shift);

    parray_l[x] = { std::norm(carray_l[x]) / sqrt(2.0 * nFrames) / nFrames, std::arg(carray_l[x]) };
    parray_r[x] = { std::norm(carray_r[x]) / sqrt(2.0 * nFrames) / nFrames, std::arg(carray_r[x]) };
  }


  // Shifting Part

  /// FFT approach
  
  const float freqoffsets = 2.0 * PI / nFrames;
  const float normfactor = 2.0 / nFrames;
  bool out_of_buf;
  
  for (int frame = 0; frame < nFrames; frame++) {
    out_of_buf = (frame < nFrames / shift);
    out_of_buf = true;
    if (out_of_buf) {
      outputs[0][frame] = 0.5 * carray_l[0].real();
      outputs[1][frame] = 0.5 * carray_r[0].real();
    }
    else {
      outputs[0][frame] = 0.0;
      outputs[1][frame] = 0.0;
    }
      float arg;
    for (int x = 1; x < nFrames / 2; x++) {
      cur_freq = Math36::getFred(frame, nFrames, sampleRate);
      if (out_of_buf) {
        arg = freqoffsets * x * frame;

        outputs[0][frame] += carray_l[x].real() *cos(arg) - carray_l[x].imag() * sin(arg);
        outputs[1][frame] += carray_r[x].real() *cos(arg) - carray_r[x].imag() * sin(arg);
      }
      else {
        arg = freqoffsets * x * frame * shift;

        outputs[0][frame] += parray_l[x].norm * cos(arg + parray_l[x].arg);
        outputs[1][frame] += parray_r[x].norm * cos(arg + parray_r[x].arg);
      }
    }
    if (out_of_buf) {
      outputs[0][frame] *= normfactor;
      outputs[1][frame] *= normfactor;
      /*for(int i=0;i<2;i++){
        if (outputs[i][frame] > 1.0 || outputs[i][frame] < -1.0)
          outputs[i][frame] = 0.0;
      } */
    }
  }
  
  /*
  for (int frame = 0; frame < nFrames; frame++) {
    float arg = shift * frame * 0.01;
    outputs[0][frame] = carray_l[frame].real()*cos(arg) - carray_l[frame].imag() * sin(arg);
    outputs[1][frame] = carray_r[frame].real() *cos(arg) - carray_r[frame].imag() * sin(arg);
  }
  */
  /// Hilbert approach


}

void IPlugEffect::GrohFreqShift(sample** inputs, sample** outputs, int nFrames)
{
  double ref = 1000, tmp;
  qso1.setFreq(ref);
  qso2.setFreq(ref + shift);
  qso1.reset();
  qso2.reset();

  for (int i = 0; i < nFrames; i++) {
    for (int j = 0; j < 2; j++) {
      tmp = inputs[j][i] * qso1.cos(); // *cos(arg + freq_prev_arg[0]);
      filter1[j] = filter1[j] + R * (tmp - filter1[j]);
      last_in1[j] = tmp;
      tmp = filter1[j] * qso2.cos(); // cos(arg * shift + freq_prev_arg[1]);
      outputs[j][i] = tmp;

      tmp = inputs[j][i] * qso1.sin();// * sin(arg + freq_prev_arg[0]);
      filter2[j] = filter2[j] + R * (tmp - filter2[j]);
      last_in2[j] = tmp;
      tmp = filter2[j] * qso2.sin();// * sin(arg * shift + freq_prev_arg[1]);
      outputs[j][i] += tmp;
    }
    qso1.next();
    qso2.next();
  }
}

void IPlugEffect::SingleSSB(sample** inputs, sample** outputs, int nFrames)
{
  double ref = 1000;
  qso1.setFreq(shift);
  qso1.reset();

  double arg, up, down, tmp;

  if (shiftChanged) {
    freq_prev_arg[0] = 0.0;
  }

  for (int frame = 0; frame < nFrames; frame++) {
    carray_l[frame] = Complex((double)inputs[0][frame], 0.0);
    carray_r[frame] = Complex((double)inputs[1][frame], 0.0);
  }

  Math36::hilbert(carray_l, nFrames);
  Math36::hilbert(carray_r, nFrames);

  for (int i = 0; i < nFrames; i++) {

    up = qso1.sin() * carray_l[i].imag();
    down = qso1.cos() * carray_l[i].real();
    tmp = down - up;
    down = up + down;
    up = tmp;
    outputs[0][i] = up;

    up = qso1.sin() * carray_r[i].imag();
    down = qso1.cos() * carray_r[i].real();
    tmp = down - up;
    down = up + down;
    up = tmp;
    outputs[1][i] = up;
    
    qso1.next();
  }
  /*
  for (int i = 0; i < 2; i++) {
    last_input[i] = outputs[i][nFrames - 1];
  }

  freq_prev_arg[0] = freqoffsets * double(nFrames) * (shift - 1.0) + freq_prev_arg[0];
  freq_prev_arg[0] = freq_prev_arg[0] - double(int(double(freq_prev_arg[0] / 2.0 / PI))) * 2.0 * PI;
  */
}

void IPlugEffect::AllpassSSB(sample** inputs, sample** outputs, int nFrames)
{
  const double freqoffsets = 2.0 * PI / double(nFrames);
  double arg, up, down, tmp;

  if (shiftChanged) {
    freq_prev_arg[0] = 0.0;
  }

  for (int frame = 0; frame < nFrames; frame++) {
    carray_l[frame] = Complex((double)inputs[0][frame], 0.0);
    carray_r[frame] = Complex((double)inputs[1][frame], 0.0);
  }

  Math36::hilbert(carray_l, nFrames);
  Math36::hilbert(carray_r, nFrames);

  for (int i = 0; i < nFrames; i++) {
    arg = freqoffsets * double(i) * (shift - 1.0) + freq_prev_arg[0];

    up = sin(arg) * carray_l[i].imag();
    down = cos(arg) * carray_l[i].real();
    tmp = down - up;
    down = up + down;
    up = tmp;
    outputs[0][i] = up;

    up = sin(arg) * carray_r[i].imag();
    down = cos(arg) * carray_r[i].real();
    tmp = down - up;
    down = up + down;
    up = tmp;
    outputs[1][i] = up;


  }

  for (int i = 0; i < 2; i++) {
    last_input[i] = outputs[i][nFrames - 1];
  }

  freq_prev_arg[0] = freqoffsets * double(nFrames) * (shift - 1.0) + freq_prev_arg[0];
  freq_prev_arg[0] = freq_prev_arg[0] - double(int(double(freq_prev_arg[0] / 2.0 / PI))) * 2.0 * PI;
}

const std::string file_path = "E:\\Libraries\\iPlug2-master\\36Effects\\36Shift\\build-win\\test.txt";

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
          + 0.1 * sin(double(s + l * size) * 0.03) + 0.2;

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
