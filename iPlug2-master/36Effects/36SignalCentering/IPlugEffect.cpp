#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include "IControls.h"

IPlugEffect::IPlugEffect(const InstanceInfo& info)
  : Plugin(info, MakeConfig(kNumParams, kNumPrograms))
{
  GetParam(kGlide)->InitDouble("Glide", 0., 0., 2000.0, 0.01, "%");
  GetParam(kDisplayOnly)->InitBool("Display Only", false);
  for (int i = 0; i < nOffset; i++)
    offset[i] = 0.0;

#if IPLUG_EDITOR // http://bit.ly/2S64BDd
  mMakeGraphicsFunc = [&]() {
    return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS, 1.);
  };
  
  mLayoutFunc = [&](IGraphics* pGraphics) {
    pGraphics->AttachCornerResizer(EUIResizerMode::Scale, false);
    pGraphics->AttachPanelBackground(IColor(255, 255, 200, 100));
    pGraphics->LoadFont("Roboto-Regular", ROBOTO_FN);
    const IRECT b = pGraphics->GetBounds();

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
     true, // Show label
     true, // Show value
     {
       IColor(0, 255, 200, 100), // Background
       IColor(255, 255, 240, 230), // Foreground
       DEFAULT_PRCOLOR, // Pressed
       COLOR_BLACK, // Frame
       DEFAULT_HLCOLOR, // Highlight
       DEFAULT_SHCOLOR, // Shadow
       IColor(255, 255, 0, 0), // Extra 1
       DEFAULT_X2COLOR, // Extra 2
       DEFAULT_X3COLOR  // Extra 3
     }, // Colors
     IText(16.f, EAlign::Center) // Label text
    };

    const IVStyle style3{
     true, // Show label
     true, // Show value
     {
       IColor(0, 255, 200, 100), // Background
       IColor(255, 255, 240, 230), // Foreground
       DEFAULT_PRCOLOR, // Pressed
       COLOR_BLACK, // Frame
       DEFAULT_HLCOLOR, // Highlight
       DEFAULT_SHCOLOR, // Shadow
       IColor(255, 0, 0, 255), // Extra 1
       DEFAULT_X2COLOR, // Extra 2
       DEFAULT_X3COLOR  // Extra 3
     }, // Colors
     IText(16.f, EAlign::Center) // Label text
    };

    const IVStyle style4{
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
    const IVStyle style5{
     true, // Show label
     true, // Show value
     {
       IColor(0, 255, 200, 100), // Background
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

    pGraphics->AttachControl(new IVSliderControl(b.GetFromLeft(60.f), kGlide, "vcontrol", style1));
    pGraphics->AttachControl(new IVDisplayControl(b.GetFromRight(PLUG_WIDTH - 60.f), "", style4, EDirection::Horizontal, -0.2, 0.2, 0.1, 1024), kNumParams+1, "LFO");
    pGraphics->AttachControl(new IVDisplayControl(b.GetFromRight(PLUG_WIDTH - 60.f), "", style5, EDirection::Horizontal, -0.2, 0.2, -0.1, 1024), kNumParams, "LFO");
    pGraphics->AttachControl(new IVDisplayControl(b.GetFromRight(PLUG_WIDTH - 60.f), "", style2, EDirection::Horizontal, -0.2, 0.2, 0.f, 1024), kOffsetL, "LFO");
    pGraphics->AttachControl(new IVDisplayControl(b.GetFromRight(PLUG_WIDTH - 60.f), "", style3, EDirection::Horizontal, -0.2, 0.2, 0.f, 1024), kOffsetR, "LFO");

    pGraphics->AttachControl(new IVSlideSwitchControl(b.GetFromRight(PLUG_WIDTH - 60.f).GetFromLeft(70.f).GetFromTop(20.f), kDisplayOnly, "Display Only", DEFAULT_STYLE.WithShowValue(false).WithShowLabel(false).WithWidgetFrac(0.5f).WithDrawShadows(false), false), kNoTag, "LFO")->SetAnimationEndActionFunction([pGraphics](IControl* pControl) {
      bool sync = pControl->GetValue() > 0.5;
      pGraphics->HideControl(kGlide, sync);
      });
  };
#endif
}

void IPlugEffect::OnIdle()
{
  mRLSender.TransmitData(*this);
}

#if IPLUG_DSP
void IPlugEffect::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  const double glide = GetParam(kGlide)->Value() / 100.;
  const int nChans = NOutChansConnected();
  double targSample = nFrames;
  targSample *= glide;
  double mod = 0.0, curoffset=0.0;
  bool dispOnly = GetParam(kDisplayOnly)->Value();

  double sum[nOffset];

    for (int c = 0; c < nChans; c++) {
      if (c < nOffset) {
        sum[c] = 0;
        for (int s = 0; s < nFrames; s++) {
          sum[c] += inputs[c][s];
        }
        sum[c] /= nFrames;
      }

      for (int s = 0; s < nFrames; s++) {
        if (c < nOffset && !dispOnly) {
          if (s < targSample)
            mod = 1.0 - (targSample - double(s)) / (targSample);
          else mod = 1.0;

          curoffset = mod * sum[c] + (1.0 - mod) * offset[c];
          outputs[c][s] = inputs[c][s] - curoffset;
        }
        else{
          curoffset = sum[c];
          outputs[c][s] = inputs[c][s];
        }
      }

      if (c < nOffset)
        offset[c] = curoffset;
      
    }

  

  mRLSender.PushData({ kOffsetL, {float(offset[0]+0.1)}});
  mRLSender.PushData({ kOffsetR, {float(offset[1]-0.1)}});
}
#endif
