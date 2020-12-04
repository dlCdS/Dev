#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include <iostream>
#include <fstream>

 const std::string file_path = "E:\\Libraries\\iPlug2-master\\36Effects\\36Compressor\\build-win\\";
// const std::string file_path = "C:\\Users\\e_fdenis\\Documents\\Dev\\iPlug2-master\\36Effects\\36Compressor\\build-win\\";

const std::string test_file = file_path + "test.txt";
const std::string test_file2 = file_path + "test2.txt";


  double getStatR(const double& x, IPlugEffect* plug) {
    return plug->buffer[0][int(x * (double) maxScopeBuffSize)];
  }

  double getStatL(const double& x, IPlugEffect* plug) {
    return plug->buffer[1][int(x * (double) maxScopeBuffSize)];
  }



IPlugEffect::IPlugEffect(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPrograms)), UIClosed(true), displayCount(0), isInit(false),
atStartCount(0), start(0.0), size(1.0)
{
  for (int s = 0; s < maxScopeBuffSize; s++) {
    for (int i = 0; i < 2; i++) {
      buffer[i][s] = 0.5;
    }
  }



  GetParam(dBpm)->InitDouble("bpm", 210, 50, 300, 0.001);
  //GetParam(limiterType)->InitEnum("Limiter type", 0, 2, "", IParam::kFlagsNone, "", "Mr", "Tr");
  // GetParam(dGrid)->InitEnum("Rate", 8, { LFO_TEMPODIV_VALIST });
  GetParam(dGrid)->InitEnum("LFO Rate", LFO<>::k1, { LFO_TEMPODIV_VALIST });
  GetParam(dStart)->InitDouble("start", 0., 0., 1., 0.001);
  GetParam(dSize)->InitDouble("size", 1., 0., 1., 0.001);
  GetParam(dZoom)->InitDouble("zoom", 1., 0., 30, 0.001);


#if IPLUG_EDITOR // http://bit.ly/2S64BDd
  mMakeGraphicsFunc = [&]() {
    return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS, 1.);
  };
  
  mLayoutFunc = [this](IGraphics* pGraphics) {
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
    
    

      pGraphics->AttachControl(new ICaptionControl(cell(0, 0).GetMidVPadded(buttonSize), dBpm, IText(24.f), IColor(255, 255, 200, 100), false), kNoTag, "vcontrols");
      // pGraphics->AttachControl(new ICaptionControl(sameCell().SubRectVertical(4, 1).GetMidVPadded(10.f), kParamGain, IText(24.f), DEFAULT_FGCOLOR, false), kNoTag, "misccontrols");
      pGraphics->AttachControl(new IVKnobControl(cell(0, 1).GetMidVPadded(buttonSize), dGrid, "grid", style, false), kNoTag, "vcontrols");

      pGraphics->AttachControl(new IVKnobControl(cell(0, 2).GetMidVPadded(buttonSize), dStart, "start", style, false), kNoTag, "vcontrols");
      pGraphics->AttachControl(new IVKnobControl(cell(0, 3).GetMidVPadded(buttonSize), dSize, "size", style, false), kNoTag, "vcontrols");
      pGraphics->AttachControl(new IVKnobControl(cell(0, 4).GetMidVPadded(buttonSize), dZoom, "zoom", style, false), kNoTag, "vcontrols");



      // Curve box
      pGraphics->AttachControl(new IVPlotControl(cell(1, 0).Union(cell(4, 5)), {
                                                              {COLOR_BLUE, [&](double x) { return getStatR(x, this); } },
                                                              {COLOR_RED, [&](double x) { return getStatL(x, this); } }


        }, 32, "IVPlotControl", style.WithShowLabel(false).WithDrawFrame(true)), kNoTag, "vcontrols");
      // Sliders
      int gfromr = 2;


       

    // Left
   
  };

  UIClosed = false;

  // file2 << "started" << std::endl;
  // if(!mLatencyDelay)
    // file2 << "no latency delay" << std::endl;
  if(testPlugin)
    testPlug();
#endif
}

#if IPLUG_DSP


void IPlugEffect::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  const int nChans = NOutChansConnected();
  const double sampleRate = GetSampleRate();

  int samplesPerBeat = (int)GetSamplesPerBeat();
  int samplePos = (int)GetSamplePos();

  

  double bpm = GetParam(dBpm)->Value(),
    grid = TempoDivisonToDouble[(int)GetParam(dGrid)->Value()];
  double zoom = GetParam(dZoom)->Value();
  
  start = GetParam(dStart)->Value();
  size = GetParam(dSize)->Value();

  if (!UIClosed)
    if (atStartCount < 100) atStartCount++;
    else if (displayCount % 1 == 0) GetUI()->SetAllControlsDirty();

  double relpos, min_relpos(start), max_relpos(start + (1.0 - start) * size);
  
  if (nChans == 2) {
    for (int s = 0; s < nFrames; s++) {
      for (int i = 0; i < nChans; i++) {
        relpos = double((samplePos + s) % int(grid * samplesPerBeat)) / grid / samplesPerBeat;
        if (relpos >= min_relpos && relpos < max_relpos) {
          buffer[i][int((relpos - min_relpos)/(max_relpos-min_relpos) * (double) maxScopeBuffSize)] = outputs[i][s]*zoom;
        }
        outputs[i][s] = inputs[i][s];

      }
    }
  } else { // Mono

  }

  mOutSender.ProcessBlock(outputs, nFrames, kCtrlTagOutput);
  //mLimSender.ProcessBlock(outputs, nFrames, kCtrlTagLim);

  isInit = true;
}

const void IPlugEffect::updateDisplay(double& var, EParams param, bool & update)
{
  double old = var;
  var = GetParam(param)->Value();
  if (var != old) update = true;
}

void IPlugEffect::dcBlock(sample** inputs)
{
}

void IPlugEffect::OnIdle()
{
  mOutSender.TransmitData(*this);
  mLimSender.TransmitData(*this);
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
  UIClosed = false;
}


void IPlugEffect::testPlug()
{
  std::fstream file(test_file, std::ios::out | std::ios::trunc);

  file.close();
  system("pause");
}
void IPlugEffect::updateCurve()
{
}
#endif
