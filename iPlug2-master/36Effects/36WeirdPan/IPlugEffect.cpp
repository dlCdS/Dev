#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include <iostream>

IPlugEffect::IPlugEffect(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPrograms))
{
  GetParam(kSpeakDelay)->InitDouble("Delay", 0.5, 0.0, 1.0, 0.01);
  GetParam(kSpeakAtt)->InitDouble("Damp", 0.15, 0.0, 1.0, 0.01);
  GetParam(kLtresh)->InitDouble("Limiter tresh", 1.0, 0.0, 1.0, 0.01);
  GetParam(kEarsDist)->InitDouble("Ears Distance - legacy", 8., 0.0, 100.0, 0.1, "cm");


  for (int i = 0; i < maxBuffSize; i++) {
    last_out_buffer[0][i] = 0;
    last_out_buffer[1][i] = 0;
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

    const int nRows = 1;
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


    
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kSpeakDelay, "Delay", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kEarsDist, "Ears Dist - leg", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kSpeakAtt, "Damp", style, false), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVKnobControl(nextCell().GetMidVPadded(buttonSize), kLtresh, "Limiter", style, false), kNoTag, "vcontrols");

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


  double s_delay = GetParam(kSpeakDelay)->Value();
  s_delay = GetParam(kEarsDist)->Value() / soundSpeed * s_delay;
  double s_att = 1.0 - GetParam(kSpeakAtt)->Value();
  int s_padding = (double)(s_delay * sampleRate);
  double tresh = GetParam(kLtresh)->Value();

  if (s_padding >= nFrames) s_padding = nFrames - 1;


  if (nChans == 2) { // Stereo Signal
    

      for (int s = 0; s < nFrames; s++) {
        outputs[0][s] = inputs[0][s] - s_att * chooseBuffer(outputs, last_out_buffer, 1, s, s_padding, nFrames);
        outputs[1][s] = inputs[1][s] - s_att * chooseBuffer(outputs, last_out_buffer, 0, s, s_padding, nFrames);

        outputs[0][s] = limiter.processSample(outputs[0][s], tresh);
        outputs[1][s] = limiter.processSample(outputs[1][s], tresh);
      }

      for (int s = 0; s < nFrames; s++) {
        last_out_buffer[0][s] = outputs[0][s];
        last_out_buffer[1][s] = outputs[1][s];
      }

  } else for (int s = 0; s < nFrames; s++) { // Mono Signal
    for (int c = 0; c < nChans; c++) {
      outputs[c][s] = inputs[c][s] ;
    }
  }
}

Limiter::Limiter() : diff(NULL), type(TRUNC), sender(NULL)
{
}

Limiter::~Limiter()
{
  for (int i = 0; i < nChans; i++)
    delete[] diff[i];
  delete[] diff;
}

void Limiter::ProcessBlock(sample** toProcess, const sample& thresh, const int& nFrames, const int& from, const int& buffSize)
{
  double tmp;
  int id;
  for (int s = 0; s < nFrames; ++s) {
    id = (from + s) % buffSize;
    for (int i = 0; i < nChans; i++) {
      tmp = toProcess[i][id];
      diff[i][s] = processSample(toProcess[i][id], thresh);
    }
  }
  if (diff != NULL && sender != NULL)
    sender->ProcessBlock(diff, nFrames, cTag);
}

void Limiter::ProcessBlock(sample toProcess[2][maxBuffSize], const sample& thresh, const int& nFrames, const int& from, const int& buffSize)
{
  double tmp;
  int id;
  for (int s = 0; s < nFrames; ++s) {
    id = (from + s) % buffSize;
    for (int i = 0; i < nChans; i++) {
      tmp = toProcess[i][id];
      diff[i][s] = processSample(toProcess[i][id], thresh);
    }
  }
  if (diff != NULL && sender != NULL)
    sender->ProcessBlock(diff, nFrames, cTag);
}

sample Limiter::processSample(sample& smp, const sample& thresh)
{
  sample out = smp;
  if (smp > thresh) {
    smp = algo(smp, thresh);

  }
  else if (smp < -1.0 * thresh) {
    smp = -1.0 * algo(-1.0 * smp, thresh);
  }
  return out;
}

void Limiter::setChannels(const int& n)
{
  nChans = n;
  diff = new sample * [nChans];
  for (int i = 0; i < nChans; i++)
    diff[i] = new sample[maxBuffSize];
}

sample Limiter::algo(const sample& input, const sample& thresh)
{
  int band;
  switch (type) {
  case TRUNC:
    return thresh;
    break;
  case MIRROR:
    band = (input - thresh) / (2.0 * thresh);
    if (band % 2 == 0)  // descending band
      return thresh - (input - thresh - 2.0 * band * thresh);
    else
      return -thresh + (input - thresh - 2.0 * band * thresh);
    break;
  default:
    return input;
    break;
  }
}

void Limiter::setType(const LType& ltype)
{
  type = ltype;
}

void Limiter::setPeakSender(IPeakSender<2>* psender, const int& ctrlTag)
{
  sender = psender;
  cTag = ctrlTag;
}


#endif