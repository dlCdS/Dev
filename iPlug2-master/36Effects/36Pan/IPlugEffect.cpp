#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include <iostream>

IPlugEffect::IPlugEffect(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPrograms))
{
  GetParam(kPan)->InitDouble("Pan", 0., -1.0, 1.0, 0.01);
  GetParam(kEarsDist)->InitDouble("Ears Distance - legacy", 8., 0.0, 100.0, 0.1, "cm");
  GetParam(kRLDist)->InitDouble("Left/Right Distance - legacy", 1., 0.0, 1.0, 0.1);
  GetParam(kPronon)->InitDouble("Prononciation", 0., -1.0, 1.0, 0.01);
  GetParam(kMonoComp)->InitDouble("Mono Correction", 0., 0.0, 1.0, 0.01);
  GetParam(kMonoLR)->InitDouble("Mono C L/R", 0.5, 0.0, 1.0, 0.01);
  GetParam(kMsType)->InitEnum("Type", 0, 3, "", IParam::kFlagsNone, "", "None", "Side", "M/S");
  GetParam(kMsCenterSide)->InitDouble("M/S C/S", 0.5, 0.0, 1.0, 0.01);


  for (int i = 0; i < maxBuffSize; i++) {
    last_buffer[0][i] = 0;
    last_buffer[1][i] = 0;
    last_ms_s[0][i] = 0;
    last_ms_s[1][i] = 0;
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


    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kPan, "Pan", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVSlideSwitchControl(nextCell().GetMidVPadded(buttonSize), kMsType, "M/S type", style, true), kNoTag, "vcontrols")->SetAnimationEndActionFunction([pGraphics](IControl* pControl) {
      bool sync = pControl->GetValue() <= 0.5;
      pGraphics->HideControl(kMsCenterSide, sync);
      });
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kMonoComp, "Mono Correct", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kRLDist, "R/L Dist - leg", style, false), kNoTag, "vcontrols");

    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kPronon, "Pronon", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kMsCenterSide, "M/S C/S", style, false), kNoTag, "vcontrols")->Hide(GetParam(kMsType)->Value()!=2);
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kMonoLR, "Mono L/R", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kEarsDist, "Ears Dist - leg", style, false), kNoTag, "vcontrols");

  };
#endif
}

#if IPLUG_DSP

const sample& chooseBuffer(sample** inputs, sample last_buffer[2][maxBuffSize], const int &lr,
  const int& s, const int &padding, const int & nFrames) {
  if (s - padding < 0) return last_buffer[lr][nFrames + s - padding];
  else if (s < nFrames) return inputs[lr][s - padding];
}

const sample& chooseBuffer(sample buffer[2][maxBuffSize], sample last_buffer[2][maxBuffSize], const int& lr,
  const int& s, const int& padding, const int& nFrames) {
  if (s - padding < 0) return last_buffer[lr][nFrames + s - padding];
  else if (s < nFrames) return buffer[lr][s - padding];
}

void IPlugEffect::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  const int nChans = NOutChansConnected();
  const double sampleRate = GetSampleRate();

  double pan = GetParam(kPan)->Value();

  double pro = GetParam(kPronon)->Value();
  double delay = GetParam(kEarsDist)->Value() / soundSpeed * pan;

  double srcDist = (1.0 - GetParam(kRLDist)->Value())/2.0;
  double gain[2] = { 1.0, 1.0 };
  double panmult(1.0);
  if (pro < 0) {
    gain[1] += pro;
  } else gain[0] -= pro;

  int MS_type = GetParam(kMsType)->Value();
  double MS_CS = GetParam(kMsCenterSide)->Value();
  double monoComp = GetParam(kMonoComp)->Value();
  double monoLR = 1.0 - GetParam(kMonoLR)->Value();
  if (pan < 0) monoLR = 1.0 - monoLR;

  int padding = (double)(delay * sampleRate);
  int id1(1), id2(0);



  if (padding <= 0) {
    padding *= -1;
    id1 = 0; id2 = 1;
  }
  if (padding >= nFrames) padding = nFrames - 1;


  if (nChans == 2) { // Stereo Signal
    // NO MS
    if(MS_type == 0) {
        
        for (int s = 0; s < nFrames; s++) {
          outputs[id2][s] = gain[id2] *
            ( (1.0 - srcDist) * chooseBuffer(inputs, last_buffer, id2, s, padding, nFrames)
            + srcDist * chooseBuffer(inputs, last_buffer, id1, s, padding, nFrames));

          outputs[id1][s] = gain[id1] * ((1.0 - srcDist) * inputs[id1][s] + srcDist * inputs[id2][s]);

          last_buffer[0][s] = inputs[0][s];
          last_buffer[1][s] = inputs[1][s];
        }

        for (int s = 0; s < nFrames; s++) {
          last_buffer[0][s] = inputs[0][s];
          last_buffer[1][s] = inputs[1][s];
        }
      }

    // Side
    if (MS_type == 1) {
      for (int s = 0; s < nFrames; s++) {
        outputs[0][s] = (gain[id1] * inputs[id2][s] + gain[id2] * chooseBuffer(inputs, last_buffer, id1, s, padding, nFrames))/2.0;
        outputs[1][s] = (gain[id1] * inputs[id1][s] + gain[id2] * chooseBuffer(inputs, last_buffer, id2, s, padding, nFrames))/2.0;
      }

      for (int s = 0; s < nFrames; s++) {
        last_buffer[0][s] = inputs[0][s];
        last_buffer[1][s] = inputs[1][s];
      }
    }

    // Hard MS
    if (MS_type == 2) {
      for (int s = 0; s < nFrames; s++) {
        ms_m[s] = (inputs[0][s] + inputs[1][s]) / 2.0;
        ms_s[0][s] = (2.0 * inputs[id1][s] - inputs[id2][s]) / 3.0;
        ms_s[1][s] = (2.0 * inputs[id2][s] - inputs[id1][s]) / 3.0;

        outputs[0][s] = (1.0 - MS_CS) * ms_m[s] + MS_CS * (gain[id1] * ms_s[0][s] + gain[id2] * chooseBuffer(ms_s, last_ms_s, 1, s, padding, nFrames));
        outputs[1][s] = (1.0 - MS_CS) * ms_m[s] + MS_CS * (gain[id1] * ms_s[1][s] + gain[id2] * chooseBuffer(ms_s, last_ms_s, 0, s, padding, nFrames));
      }

      for (int s = 0; s < nFrames; s++) {
        last_ms_s[0][s] = ms_s[0][s];
        last_ms_s[1][s] = ms_s[1][s];
      }
    }

    // Mono Compensation
    sample mono_in, mono_out, delta;
    for (int s = 0; s < nFrames; s++) {
      mono_in = inputs[0][s] + inputs[1][s];
      mono_out = outputs[0][s] + outputs[1][s];
      delta = mono_out - mono_in;
      outputs[0][s] += monoComp * monoLR * delta;
      outputs[1][s] += monoComp * (1.0 - monoLR) * delta;
    } 

  } else for (int s = 0; s < nFrames; s++) { // Mono Signal
    for (int c = 0; c < nChans; c++) {
      outputs[c][s] = inputs[c][s] ;
    }
  }
}

#endif