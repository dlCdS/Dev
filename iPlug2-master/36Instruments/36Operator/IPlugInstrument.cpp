#include "IPlugInstrument.h"
#include "IPlug_include_in_plug_src.h"
#include "LFO.h"

double getPreEffectFFT(const double& x, VirtualShapeGenerator* vsg) {
  return vsg->GetPreEffectFFT(20+22000 * pow(x, 4)) + 0.2;
}

double getPostEffectFFT(const double& x, VirtualShapeGenerator* vsg) {
  return vsg->GetPostEffectFFT(20+22000 * pow(x, 4));
}

double getAttackFilter(const double& x, VirtualShapeGenerator* vsg) {
  return 4.0 * vsg->GetAttackFilter(20 + 22000 * pow(x, 4));
}

double getLFOFilter(const double& x, VirtualShapeGenerator* vsg) {
  return 4.0 * vsg->GetLFOFilter(20 + 22000 * pow(x, 4));
}

double getEnvelope(const double& x, VirtualShapeGenerator* vsg) {
  return vsg->GetEnvelope(x);
}

IPlugInstrument::IPlugInstrument(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets)),
UIClosed(true)
{

  GetParam(kParamGain)->InitDouble("Gain", 20., 0., 100.0, 0.01, "%");

  GetParam(kParamNoteGlideTime)->InitDouble("Glide Time", 0., 0.0, 10000., 1., "ms");
  GetParam(kParamDoGlide)->InitBool("Glide", true);
  GetParam(kParamVoices)->InitInt("Voices", 8, 1, 32);

  GetParam(kParamAttack)->InitDouble("Attack", 10., 1., 10000., 1., "ms", IParam::kFlagsNone, "ADSR", IParam::ShapePowCurve(3.));
  GetParam(kParamDecay)->InitDouble("Decay", 10., 1., 10000., 1., "ms", IParam::kFlagsNone, "ADSR", IParam::ShapePowCurve(3.));
  GetParam(kParamSustain)->InitDouble("Sustain", 50., 0., 100., 1, "%", IParam::kFlagsNone, "ADSR");
  GetParam(kParamRelease)->InitDouble("Release", 10., 2., 10000., 1., "ms", IParam::kFlagsNone, "ADSR");


  GetParam(kPitchAttack)->InitBool("Pitch Attack", true);
  GetParam(kPitchAttackValue)->InitDouble("PA Value", 0., -24., 24., 0.05);
  GetParam(kPitchAttackTime)->InitDouble("PA Time", 10., 0.0, 10000., 1., "ms");
  
  GetParam(kParamPitchBend)->InitDouble("Bend", 0., -12., 12., .01);

  GetParam(kParamShape)->InitEnum("Shape", LFO<>::kTriangle, { LFO_SHAPE_VALIST });
  GetParam(kParamPhase)->InitPercentage("Phase");
  GetParam(kParamPitch)->InitInt("Pitch", 0, -24, 24);
  GetParam(kParamDetune)->InitDouble("Detune", 0., -1, 1., .001);
  GetParam(kParamSpread)->InitDouble("Spread", 1., 1., 1.05, 0.001);


  GetParam(kParamLFOFilterShape)->InitEnum("LFO Shape", LFO<>::kTriangle, {LFO_SHAPE_VALIST});
  GetParam(kParamLFOFilterRateHz)->InitFrequency("LFO Rate", 1., 0.01, 40.);
  GetParam(kParamLFOFilterRateTempo)->InitEnum("LFO Rate", LFO<>::k1, {LFO_TEMPODIV_VALIST});
  GetParam(kParamLFOFilterRateMode)->InitBool("LFO Sync", true);
  GetParam(kParamLFOFilterPhase)->InitPercentage("LFO Depth");
  GetParam(kParamLFOFilter)->InitPercentage("LFO Depth");

  GetParam(kLFOFilterType)->InitEnum("Filter Type", 0, { "Lowpass", "Highpass", "Bandpass" });



  GetParam(kFilterAttackFromF)->InitDouble("FA From", 0., 0., 22000, 1., "Hz");
  GetParam(kFilterAttackToF)->InitDouble("FA To", 0., 0., 22000, 1., "Hz");

  GetParam(kFilterAttackFromN)->InitDouble("FA From N", 0., -3000, 10000, 1., "Hz");
  GetParam(kFilterAttackToN)->InitDouble("FA To N", 0., -3000., 10000., 1., "Hz");
  GetParam(kFilterAttack)->InitBool("Filter Attack", false);
  GetParam(kFilterAttackType)->InitEnum("FA Type", 0, { "Lowpass", "Highpass", "Bandpass" });
  GetParam(kFilterAttackFollow)->InitEnum("FA Note Follow", 0, { "No", "Yes" });
  GetParam(kFilterAttackTrig)->InitEnum("FA Trigger", 0, { "Once", "Always" });
  GetParam(kFilterAttackResFrom)->InitDouble("FA Res From", 0., 0., 1., 0.001);
  GetParam(kFilterAttackResTo)->InitDouble("FA Res To", 0., 0., 1., 0.001);
  GetParam(kFilterAttackTime)->InitDouble("FA Time", 0., 0., 10000., 10., "ms");
  GetParam(kFilterAttackAtt)->InitEnum("FA Att", 0, { "-12", "-18", "-24" });
    
    
    

  GetParam(kParamLFOGainShape)->InitEnum("LFO Shape", LFO<>::kTriangle, { LFO_SHAPE_VALIST });
  GetParam(kParamLFOGainRateHz)->InitFrequency("LFO Rate", 1., 0.01, 40.);
  GetParam(kParamLFOGainRateTempo)->InitEnum("LFO Rate", LFO<>::k1, { LFO_TEMPODIV_VALIST });
  GetParam(kParamLFOGainRateMode)->InitBool("LFO Sync", true);
  GetParam(kParamLFOGainPhase)->InitPercentage("LFO Depth");
  GetParam(kParamLFOGain)->InitPercentage("LFO Depth");
  
#if IPLUG_EDITOR // http://bit.ly/2S64BDd
  mMakeGraphicsFunc = [&]() {
    return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS, GetScaleForScreen(PLUG_WIDTH, PLUG_HEIGHT));
  };
  
  mLayoutFunc = [&](IGraphics* pGraphics) {
    pGraphics->AttachCornerResizer(EUIResizerMode::Scale, false);
    pGraphics->AttachPanelBackground(COLOR_GRAY);
    pGraphics->EnableMouseOver(true);
    pGraphics->EnableMultiTouch(true);
    
#ifdef OS_WEB
    pGraphics->AttachPopupMenuControl();
#endif

    const float buttonSize = 45.f;
    const float buttonUp = 25.f;
    const float frameUp = 20.f;



    const IRECT b = pGraphics->GetBounds().GetPadded(-5.f);

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

    auto cell = [&](int r, int c) {
      return b.GetGridCell(r * PLUG_COLS + c, PLUG_ROWS, PLUG_COLS).GetPadded(-5.);
    };

//    pGraphics->EnableLiveEdit(true);
    pGraphics->LoadFont("Roboto-Regular", ROBOTO_FN);
    const IRECT lfoPanel = b.GetFromLeft(300.f).GetFromTop(200.f);

    // KEYBOARD -------
    pGraphics->AttachControl(new IVKeyboardControl(cell(PLUG_ROWS - 2, 5).Union(cell(PLUG_ROWS - 1, PLUG_COLS - 1)), kCtrlTagKeyboard));
    pGraphics->AttachControl(new IWheelControl(cell(PLUG_ROWS-2, 4).Union(cell(PLUG_ROWS-1, 4)).GetMidHPadded(20)), kCtrlTagBender);

    /// MAIN ---------
      pGraphics->AttachControl(new IVKnobControl(cell(0, PLUG_COLS - 5).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamShape, "Shape"), kNoTag, "Main");
      pGraphics->AttachControl(new IVKnobControl(cell(1, PLUG_COLS - 5).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamPhase, "Phase"), kNoTag, "Main");
      pGraphics->AttachControl(new IVKnobControl(cell(2, PLUG_COLS - 5).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamPitch, "Pitch"), kNoTag, "Main");
      pGraphics->AttachControl(new IVKnobControl(cell(3, PLUG_COLS - 5).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamDetune, "Detune"), kNoTag, "Main");
      pGraphics->AttachControl(new IVKnobControl(cell(4, PLUG_COLS - 5).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamSpread, "Spread"), kNoTag, "Main");
      pGraphics->AttachControl(new IVGroupControl("Main", "Main", 0.0, frameUp, 0.0, 0.0));

    /// GAIN & METER --------
    pGraphics->AttachControl(new IVKnobControl(cell(0, PLUG_COLS - 1).GetMidVPadded(buttonSize), kParamGain, "Gain"));
    pGraphics->AttachControl(new IVLEDMeterControl<2>(cell(1, PLUG_COLS - 1).Union(cell(PLUG_ROWS - 3, PLUG_COLS - 1))), kCtrlTagMeter);

    /// GLIDE --------
    pGraphics->AttachControl(new IVKnobControl(cell(3, PLUG_COLS - 2).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamVoices, "Voices"), kNoTag, "Glide");
    pGraphics->AttachControl(new IVKnobControl(cell(3, PLUG_COLS - 3).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamNoteGlideTime, "Glide Time"), kNoTag, "Glide")->Hide(!GetParam(kParamDoGlide));
    pGraphics->AttachControl(new IVSlideSwitchControl(cell(3, PLUG_COLS - 4).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamDoGlide, "Glide",
      DEFAULT_STYLE.WithShowValue(false).WithShowLabel(false).WithWidgetFrac(0.5f), false), kNoTag, "Glide")->SetAnimationEndActionFunction([pGraphics](IControl* pControl) {
        bool sync = pControl->GetValue() > 0.5;
        pGraphics->HideControl(kParamNoteGlideTime, !sync);
        });
    pGraphics->AttachControl(new IVGroupControl("Glide", "Glide", 0.0, frameUp, 0.0, 0.0));

    /// ENVELOPE -------
    pGraphics->AttachControl(new IVKnobControl(cell(0, PLUG_COLS - 4).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamAttack, "Attack"), kNoTag, "Env");
    pGraphics->AttachControl(new IVKnobControl(cell(0, PLUG_COLS - 3).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamDecay, "Decay"), kNoTag, "Env");
    pGraphics->AttachControl(new IVKnobControl(cell(1, PLUG_COLS - 2).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamSustain, "Sustain"), kNoTag, "Env");
    pGraphics->AttachControl(new IVKnobControl(cell(0, PLUG_COLS - 2).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamRelease, "Release"), kNoTag, "Env");
    pGraphics->AttachControl(new IVPlotControl(cell(1, PLUG_COLS - 4).Union(cell(1, PLUG_COLS- 3)), {
                                                           {IColor(255, 0, 0, 50), [&](double x) { return getEnvelope(x, &sineshape); } }


      }, 1024, "IVPlotControl", DEFAULT_STYLE.WithShowLabel(false).WithDrawFrame(true), 0.1, 1.), kNoTag, "vcontrols");
    pGraphics->AttachControl(new IVGroupControl("Envelope", "Env", 0.0, frameUp, 0.0, 0.0));


    /// PITCH ATTACK --------
    pGraphics->AttachControl(new IVKnobControl(cell(2, PLUG_COLS - 3).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kPitchAttackValue, "PA Value"), kNoTag, "PA")->Hide(!GetParam(kPitchAttack));
    pGraphics->AttachControl(new IVKnobControl(cell(2, PLUG_COLS - 2).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kPitchAttackTime, "PA Time"), kNoTag, "PA")->Hide(!GetParam(kPitchAttack));
    pGraphics->AttachControl(new IVSlideSwitchControl(cell(2, PLUG_COLS - 4).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kPitchAttack, "Pitch Attack",
      DEFAULT_STYLE.WithShowValue(false).WithShowLabel(false).WithWidgetFrac(0.5f)), kNoTag, "PA")->SetAnimationEndActionFunction([pGraphics](IControl* pControl) {
      bool sync = pControl->GetValue() > 0.5;
      pGraphics->HideControl(kPitchAttackValue, !sync);
      pGraphics->HideControl(kPitchAttackTime, !sync);
      });
    pGraphics->AttachControl(new IVGroupControl("Pitch Attack", "PA", 0.0, frameUp, 0.0, 0.0));

    /// FILTER ATTACK --------
    pGraphics->AttachControl(new IVSlideSwitchControl(cell(2, 8).GetFromBottom(buttonSize + buttonUp).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttack, "Sync",
      DEFAULT_STYLE.WithShowValue(false).WithShowLabel(false).WithWidgetFrac(0.5f), false), kNoTag, "Filter Attack")->SetAnimationEndActionFunction([&, pGraphics](IControl* pControl) {
        bool sync = pControl->GetValue() < 0.5;
        bool sync2 = this->GetParam(kFilterAttackFollow)->Value() < 1;

        pGraphics->HideControl(kFilterAttackFromF, sync || !sync2);
        pGraphics->HideControl(kFilterAttackFromN, sync || sync2);
        pGraphics->HideControl(kFilterAttackToF, sync || !sync2);
        pGraphics->HideControl(kFilterAttackToN, sync || sync2);
        pGraphics->HideControl(kFilterAttackType, sync);
        pGraphics->HideControl(kFilterAttackAtt, sync);
        pGraphics->HideControl(kFilterAttackFollow, sync);
        pGraphics->HideControl(kFilterAttackResFrom, sync);
        pGraphics->HideControl(kFilterAttackResTo, sync);
        pGraphics->HideControl(kFilterAttackTime, sync);
        pGraphics->HideControl(kFilterAttackTrig, sync);
        });

    bool sync = GetParam(kFilterAttack)->Value() < 0.5;
    bool sync2 = this->GetParam(kFilterAttackFollow)->Value() < 1;
    pGraphics->AttachControl(new IVKnobControl(cell(1, 4).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttackFromF, "From"), kNoTag, "Filter Attack")->Hide(sync || !sync2);
    pGraphics->AttachControl(new IVKnobControl(cell(1, 4).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttackFromN, "From Note"), kNoTag, "Filter Attack")->Hide(sync || sync2);
    pGraphics->AttachControl(new IVKnobControl(cell(1, 5).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttackToF, "To"), kNoTag, "Filter Attack")->Hide(sync || !sync2);
    pGraphics->AttachControl(new IVKnobControl(cell(1, 5).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttackToN, "To Note"), kNoTag, "Filter Attack")->Hide(sync || sync2);
    pGraphics->AttachControl(new IVKnobControl(cell(1, 6).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttackType, "Type"), kNoTag, "Filter Attack")->Hide(sync);
    pGraphics->AttachControl(new IVKnobControl(cell(1, 7).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttackAtt, "Attenuation"), kNoTag, "Filter Attack")->Hide(sync);
    pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 8).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttackFollow, "Follow Note",
      DEFAULT_STYLE.WithShowValue(true).WithShowLabel(true).WithWidgetFrac(0.5f), false), kNoTag, "Filter Attack")->SetAnimationEndActionFunction([&, pGraphics](IControl* pControl) {
        bool sync(pControl->GetValue() < 0.5);

      pGraphics->HideControl(kFilterAttackFromF, !sync);
      pGraphics->HideControl(kFilterAttackToF, !sync);
      pGraphics->HideControl(kFilterAttackFromN, sync);
      pGraphics->HideControl(kFilterAttackToN, sync);
      })->Hide(sync);
      pGraphics->AttachControl(new IVSlideSwitchControl(cell(2, 7).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttackTrig, "Trigger Mode",
        DEFAULT_STYLE.WithShowValue(true).WithShowLabel(true).WithWidgetFrac(0.5f), false), kNoTag, "Filter Attack")->Hide(sync);
      pGraphics->AttachControl(new IVKnobControl(cell(2, 4).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttackResFrom, "Res From"), kNoTag, "Filter Attack")->Hide(sync);
      pGraphics->AttachControl(new IVKnobControl(cell(2, 5).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttackResTo, "Res To"), kNoTag, "Filter Attack")->Hide(sync);
      pGraphics->AttachControl(new IVKnobControl(cell(2, 6).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kFilterAttackTime, "Time"), kNoTag, "Filter Attack")->Hide(sync);
      pGraphics->AttachControl(new IVGroupControl("Filter Attack", "Filter Attack", 0.0, frameUp, 0.0, 0.0));
      
      
      
    /// LFO FILTER --------
    pGraphics->AttachControl(new IVKnobControl(cell(1, 0).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOFilterRateHz, "Rate"), kNoTag, "LFO Filter")->Hide(true);
    pGraphics->AttachControl(new IVKnobControl(cell(1, 0).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOFilterRateTempo, "Rate"), kNoTag, "LFO Filter")->DisablePrompt(false);
    pGraphics->AttachControl(new IVKnobControl(cell(0, 0).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOFilterPhase, "Phase"), kNoTag, "LFO Filter");
    pGraphics->AttachControl(new IVKnobControl(cell(0, 2).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOFilter, "Freq"), kNoTag, "LFO Filter");
    pGraphics->AttachControl(new IVKnobControl(cell(0, 1).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOFilterShape, "Shape"), kNoTag, "LFO Filter")->DisablePrompt(false);
    pGraphics->AttachControl(new IVSlideSwitchControl(cell(2, 0).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOFilterRateMode, "Sync",
      DEFAULT_STYLE.WithShowValue(false).WithShowLabel(false).WithWidgetFrac(0.5f), false), kNoTag, "LFO Filter")->SetAnimationEndActionFunction([pGraphics](IControl* pControl) {
      bool sync = pControl->GetValue() > 0.5;
      pGraphics->HideControl(kParamLFOFilterRateHz, sync);
      pGraphics->HideControl(kParamLFOFilterRateTempo, !sync);
    });
    pGraphics->AttachControl(new IVDisplayControl(cell(1, 1).Union(cell(2, 3)).GetVShifted(-0), "", DEFAULT_STYLE, EDirection::Horizontal, 0.f, 1.f, 0.f, 1024), kCtrlTagLFOFilterVis, "LFO Filter");
    pGraphics->AttachControl(new IVGroupControl("LFO Filter", "LFO Filter", 0.0, frameUp, 0.0, -10.0));
     
    /// LFO GAIN --------
    pGraphics->AttachControl(new IVKnobControl(cell(4, 0).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOGainRateHz, "Rate"), kNoTag, "LFO Gain")->Hide(true);
    pGraphics->AttachControl(new IVKnobControl(cell(4, 0).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOGainRateTempo, "Rate"), kNoTag, "LFO Gain")->DisablePrompt(false);
    pGraphics->AttachControl(new IVKnobControl(cell(3, 0).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOGainPhase, "Phase"), kNoTag, "LFO Gain");
    pGraphics->AttachControl(new IVKnobControl(cell(3, 2).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOGain, "Freq"), kNoTag, "LFO Gain");
    pGraphics->AttachControl(new IVKnobControl(cell(3, 1).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOGainShape, "Shape"), kNoTag, "LFO Gain")->DisablePrompt(false);
    pGraphics->AttachControl(new IVSlideSwitchControl(cell(5, 0).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kParamLFOGainRateMode, "Sync",
      DEFAULT_STYLE.WithShowValue(false).WithShowLabel(false).WithWidgetFrac(0.5f), false), kNoTag, "LFO Gain")->SetAnimationEndActionFunction([pGraphics](IControl* pControl) {
        bool sync = pControl->GetValue() > 0.5;
        pGraphics->HideControl(kParamLFOGainRateHz, sync);
        pGraphics->HideControl(kParamLFOGainRateTempo, !sync);
        });
    pGraphics->AttachControl(new IVDisplayControl(cell(4, 1).Union(cell(5, 3)).GetVShifted(0.), "", DEFAULT_STYLE, EDirection::Horizontal, 0.f, 1.f, 0.f, 1024), kCtrlTagLFOGainVis, "LFO Gain");
    pGraphics->AttachControl(new IVGroupControl("LFO Gain", "LFO Gain", 0.0, frameUp, 0.0, -10.0));
    
    //LFO FILTER --------
    pGraphics->AttachControl(new IVKnobControl(cell(0, 6).GetFromBottom(buttonSize + buttonUp).GetMidVPadded(buttonSize), kLFOFilterType, "Type"), kNoTag, "LFO Filter");

    // FFT & filter display
    // Curve box
    pGraphics->AttachControl(new IVPlotControl(cell(3, 4).Union(cell(6, 8)), {
                                                            {IColor(255, 255, 170, 200), [&](double x) { return getPreEffectFFT(x, &sineshape); } },
                                                            {IColor(255, 0, 0, 50), [&](double x) { return getPostEffectFFT(x, &sineshape); } },
                                                            {COLOR_BLACK, [&](double x) { return getAttackFilter(x, &sineshape); } },
                                                            {COLOR_WHITE, [&](double x) { return getLFOFilter(x, &sineshape); } }


      }, 1024, "IVPlotControl", DEFAULT_STYLE.WithShowLabel(false).WithDrawFrame(true), 1.2, 11.), kNoTag, "vcontrols");
#ifdef OS_IOS
    if(!IsAuv3AppExtension())
    {
      pGraphics->AttachControl(new IVButtonControl(b.GetFromTRHC(100, 100), [pGraphics](IControl* pCaller) {
                               dynamic_cast<IGraphicsIOS*>(pGraphics)->LaunchBluetoothMidiDialog(pCaller->GetRECT().L, pCaller->GetRECT().MH());
                               SplashClickActionFunc(pCaller);
                             }, "BTMIDI"));
    }
#endif
    
    pGraphics->SetQwertyMidiKeyHandlerFunc([pGraphics](const IMidiMsg& msg) {
                                              pGraphics->GetControlWithTag(kCtrlTagKeyboard)->As<IVKeyboardControl>()->SetNoteFromMidi(msg.NoteNumber(), msg.StatusMsg() == IMidiMsg::kNoteOn);
                                           });
  };
#endif
}

#if IPLUG_DSP
void IPlugInstrument::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  sineshape.ProcessBlock(outputs, 2, nFrames, GetSamplePos(), GetSamplesPerBeat(), GetSampleRate());
  mMeterSender.ProcessBlock(outputs, nFrames, kCtrlTagMeter);
  mLFOFilterVisSender.PushData({ kCtrlTagLFOFilterVis, {(float)sineshape._lfofilter.getLastValue()} });
  mLFOGainVisSender.PushData({ kCtrlTagLFOFilterVis, {(float)sineshape._lfogain.getLastValue()} });
  mLFOPitchVisSender.PushData({ kCtrlTagLFOFilterVis, {(float)sineshape._lfopitch.getLastValue()} });
  if (!UIClosed)
    GetUI()->SetAllControlsDirty();
}

void IPlugInstrument::OnIdle()
{
  mMeterSender.TransmitData(*this);
  mLFOFilterVisSender.TransmitData(*this);
  mLFOGainVisSender.TransmitData(*this);
  mLFOPitchVisSender.TransmitData(*this);
}

void IPlugInstrument::OnReset()
{
  sineshape.Reset(GetSampleRate());
}

void IPlugInstrument::ProcessMidiMsg(const IMidiMsg& msg)
{
  TRACE;
  
  int status = msg.StatusMsg();
  
  switch (status)
  {
    case IMidiMsg::kNoteOn:
    case IMidiMsg::kNoteOff:
    case IMidiMsg::kPolyAftertouch:
    case IMidiMsg::kControlChange:
    case IMidiMsg::kProgramChange:
    case IMidiMsg::kChannelAftertouch:
    case IMidiMsg::kPitchWheel:
    {
      goto handle;
    }
    default:
      return;
  }
  
handle:
  sineshape.ProcessMidiMsg(msg);
  
}

void IPlugInstrument::OnParamChange(int paramIdx)
{
  /* Recap param
  kParamGain = 0,

  kParamNoteGlideTime,
  kParamDoGlide,

  kParamVoices,

  kParamShape,
  kParamPhase,
  kParamPitch,
  kParamDetune,
  kParamSpread,

  kParamAttack,
  kParamDecay,
  kParamSustain,
  kParamRelease,

  kParamPitchBend,

  kFilterAttack,
  kFilterAttackFromF,
  kFilterAttackFromN,
  kFilterAttackToF,
  kFilterAttackToN,
  kFilterAttackType,
  kFilterAttackAtt,
  kFilterAttackFollow,
  kFilterAttackResFrom,
  kFilterAttackResTo,
  kFilterAttackTime,

  kPitchAttack,
  kPitchAttackValue,
  kPitchAttackTime,

  kParamLFOFilterShape,
  kParamLFOFilterRateHz,
  kParamLFOFilterRateTempo,
  kParamLFOFilterRateMode,
  kParamLFOFilterPhase,
  kParamLFOFilter,

  kNumParams*/
  int val;
  switch (paramIdx) {
    /// GAIN
  case kParamGain:
    sineshape.SetGain(GetParam(paramIdx)->Value()/100.);
    break;

    /// GLIDE
  case kParamNoteGlideTime:
    sineshape.SetGlideTime(GetParam(paramIdx)->Value()/1000.);
    break;
  case kParamDoGlide:
    sineshape.EnableGlide(GetParam(paramIdx)->Value());
    break;
  case kParamVoices:
    sineshape.SetVoices(GetParam(paramIdx)->Value());
    break;

    /// MAIN
  case kParamShape:
    sineshape.setShape((Shape::Type)GetParam(paramIdx)->Value());
    break;
  case kParamPhase:
    sineshape.SetPhase(GetParam(paramIdx)->Value()/100.);
    break;
  case kParamPitch:
    sineshape.SetPitch(GetParam(paramIdx)->Value());
    break;
  case kParamDetune:
    sineshape.SetDetune(GetParam(paramIdx)->Value());
    break;
  case kParamSpread:
    sineshape.SetSpread(GetParam(paramIdx)->Value());
    break;

    /// ENVELOPE
  case kParamAttack:
    sineshape.SetAttack(GetParam(paramIdx)->Value() / 1000.);
    break;
  case kParamDecay:
    sineshape.SetDecay(GetParam(paramIdx)->Value() / 1000.);
    break;
  case kParamSustain:
    sineshape.SetSustain(GetParam(paramIdx)->Value()/100.);
    break;
  case kParamRelease:
    sineshape.SetRelease(GetParam(paramIdx)->Value() / 1000.);
    break;
  case  kParamPitchBend:
    sineshape.SetPitchBend(GetParam(paramIdx)->Value());
    break;

    /// PITCH ATTACK
  case kPitchAttack:
    sineshape.EnablePitch(GetParam(paramIdx)->Value());
    break;
  case kPitchAttackValue:
    sineshape.SetPitchAttack(GetParam(paramIdx)->Value());
    break;
  case kPitchAttackTime:
    sineshape.SetPitchTime(GetParam(paramIdx)->Value() / 1000);
    break;

    

  /// FILTER ATTACK
  case kFilterAttack:
    sineshape.EnableAttackFilter(GetParam(paramIdx)->Value());
    break;
  case kFilterAttackFromF:
    sineshape.SetAttackFilterFromF(GetParam(paramIdx)->Value());
    break;
  case kFilterAttackFromN:
    sineshape.SetAttackFilterFromN(GetParam(paramIdx)->Value());
    break;
  case kFilterAttackToF:
    sineshape.SetAttackFilterToF(GetParam(paramIdx)->Value());
    break;
  case  kFilterAttackToN:
    sineshape.SetAttackFilterToN(GetParam(paramIdx)->Value());
    break;
  case  kFilterAttackType:
    sineshape.SetAttackFilterType((Math36::Filter::FilterMode) GetParam(paramIdx)->Value());
    break;
  case kFilterAttackAtt:
    sineshape.SetAttackFilterAtt(GetParam(paramIdx)->Value());
    break;
  case  kFilterAttackFollow:
    sineshape.SetAttackFilterFollow(GetParam(paramIdx)->Value());
    break;
  case kFilterAttackTrig:
    sineshape.SetAttackFilterTrig((VirtualShapeGenerator::LaunchMode) GetParam(paramIdx)->Value());
    break;
  case  kFilterAttackResFrom:
    sineshape.SetAttackFilterResFrom(GetParam(paramIdx)->Value());
    break;
  case kFilterAttackResTo:
    sineshape.SetAttackFilterResTo(GetParam(paramIdx)->Value());
    break;
  case kFilterAttackTime:
    sineshape.SetAttackFilterTime(GetParam(paramIdx)->Value());
    break;

    /// LFO FILTER
  case kLFOFilterType:
    sineshape.SetLFOFilterType((Math36::Filter::FilterMode)GetParam(paramIdx)->Value());
    break;
  case kParamLFOFilterShape:
    sineshape._lfofilter.setType((Shape::Type)GetParam(paramIdx)->Value());
    break;
  case  kParamLFOFilterRateHz:
    sineshape._lfofilter.setFreq(GetParam(paramIdx)->Value());
    break;
  case  kParamLFOFilterRateTempo:
    sineshape._lfofilter.setRate(GetParam(paramIdx)->Value(), GetTempo());
    break;
  case kParamLFOFilterRateMode:
    sineshape._lfofilter.setMode(GetParam(paramIdx)->Value());
    break;
  case kParamLFOFilterPhase:
    sineshape._lfofilter.setPhase(GetParam(paramIdx)->Value()/100.);
    break;
  case  kParamLFOFilter:
    sineshape.SetLFOFilterMix(GetParam(paramIdx)->Value() / 100.);
    break;
  }
}

bool IPlugInstrument::OnMessage(int msgTag, int ctrlTag, int dataSize, const void* pData)
{
  if(ctrlTag == kCtrlTagBender && msgTag == IWheelControl::kMessageTagSetPitchBendRange)
  {
    const int bendRange = *static_cast<const int*>(pData);
    sineshape.SetPitchBend(bendRange);
  }
  
  return false;
}

void IPlugInstrument::OnUIClose()
{
  Plugin::OnUIClose();
  GetUI()->SetAllControlsClean();
  UIClosed = true;
  Sleep(500);
}

void IPlugInstrument::OnUIOpen()
{
  Plugin::OnUIOpen();
  UIClosed = false;
}

#endif
