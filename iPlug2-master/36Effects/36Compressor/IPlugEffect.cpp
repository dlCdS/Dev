#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include <iostream>
#include <fstream>

 const std::string file_path = "E:\\Libraries\\iPlug2-master\\36Effects\\36Compressor\\build-win\\";
// const std::string file_path = "C:\\Users\\e_fdenis\\Documents\\Dev\\iPlug2-master\\36Effects\\36Compressor\\build-win\\";

const std::string test_file = file_path + "test.txt";
const std::string test_file2 = file_path + "test2.txt";




// std::fstream file2(test_file2, std::ios::out | std::ios::trunc);



  double getSigmoid(const double& x, IPlugEffect* plug) {
    return 2.4 * plug->sigmoid.get(x) - 1.2;
  }

  double getValueDisplay(const double& x, IPlugEffect *plug) {
    if (x <= .2) return (x * 5.0) * (plug->_cp0) * 2. - 1.2;
    else if (x < .4) return ((x - .2) * 5.0 * (plug->_cp1 - plug->_cp0) + plug->_cp0) * 2. - 1.2;
    else if (x < .6) return ((x - .4) * 5.0 * (plug->_cp2 - plug->_cp1) + plug->_cp1) * 2. - 1.2;
    else if (x < .8) return ((x - .6) * 5.0 * (plug->_cp3 - plug->_cp2) + plug->_cp2) * 2. - 1.2;
    else return ((x - .8) * 5.0 * (plug->_cp4 - plug->_cp3) + plug->_cp3) * 2. - 1.2;
  }

  const double dispmut = 2.0,
    dispalign = dispmut * 0.0 + 1.1;

  double getStatR(const double& x, IPlugEffect* plug) {

    if (x <= 0.0)
      return dispmut * plug->my_stat[0][0] - dispalign;
    else if (x >= 1.0)
      return dispmut * plug->my_stat[0][9] - dispalign;
    return dispmut * plug->my_stat[0][int(x * 10.0)] - dispalign;
  }

  double getStatL(const double& x, IPlugEffect* plug) {
    if (x <= 0.0)
      return dispmut * plug->my_stat[1][0] - dispalign;
    else if (x >= 1.0)
      return dispmut * plug->my_stat[1][9] - dispalign;
    return dispmut * plug->my_stat[1][int(x * 10.0)] - dispalign;
  }



IPlugEffect::IPlugEffect(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPrograms)), UIClosed(true), displayCount(0), isInit(false),
atStartCount(0), bsendDiff(false), breplot(false), pro_pos(0)
{
  for (int s = 0; s < maxBuffSize; s++) {
    y_buffer[0][s] = 0.0;
    y_buffer[1][s] = 0.0;
  }

  for (int s = 0; s < 10; s++)
    for (int i = 0; i < 2; i++) {
      my_stat[i][s] = 0.0;
      cur_stat[i][s] = 0.0;
      for (int d = 0; d < displayLoop; d++)
        prev_stat[i][s][d] = 0.0;
    }

  limiter.setChannels(2);
  limiter.setPeakSender(&mLimSender, kCtrlTagLim);

  GetParam(dcR)->InitDouble("DC-R", .998, 0.98, .999, 0.001);
  GetParam(limiterType)->InitEnum("Limiter type", 0, 2, "", IParam::kFlagsNone, "", "Mr", "Tr");
  GetParam(threshold)->InitDouble("Threshold", 0.5, 0.0, 1.0, 0.01);
  GetParam(maxTresh)->InitDouble("Max Threshold", 1.25, 1.0, 3.0, 0.05);
  GetParam(cp0)->InitDouble("Inf+", .25, 0.0, 1.2, 0.01);
  GetParam(cp1)->InitDouble("Thresh+", 0.5, 0.0, 1.2, 0.01);
  GetParam(cp2)->InitDouble("Half+", 0.75, 0.0, 1.2, 0.01);
  GetParam(cp3)->InitDouble("Zero", 1., 0.0, 1.2, 0.01);
  GetParam(cp4)->InitDouble("Half-", 1.25, 0.0, 1.2, 0.01);
  GetParam(outgain)->InitDouble("Gain Out", 1., 0.0, 3.0, 0.01);
  GetParam(steep)->InitDouble("Stiffness", 0., -20., 20., 0.1);
  GetParam(sendDiff)->InitBool("Send Diff", false);
  GetParam(replot)->InitBool("Replot", false);
  GetParam(mix)->InitDouble("Mix", 1., 0.0, 1.0, 0.01);


  

  ldot[0] = 0;
  ldot[1] = 0;

  ldot_age[0] = 0;
  ldot_age[1] = 0;


  last_delta[0] = 0.0;
  last_delta[1] = 0.0;

  var[0] = 0.0;
  var[1] = 0.0;

  lvar[0] = -1.0;
  lvar[1] = -1.0;

  last_buffer[0] = 0.0;
  last_buffer[1] = 0.0;


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
    
    

      // upper left
      pGraphics->AttachControl(new IVKnobControl(cell(0, 3).GetMidVPadded(buttonSize), steep, "Stiffness", style, false), kNoTag, "vcontrols");
      pGraphics->AttachControl(new IVPlotControl(cell(0, 2).Union(cell(0, 2)), { {COLOR_DARK_GRAY , [&](double x) { return getSigmoid(x, this); } } }, 32, "", style), kNoTag, "vcontrols");
      pGraphics->AttachControl(new IVSlideSwitchControl(cell(0, 4).GetMidVPadded(buttonSize), replot, "Replot", style.WithShowValue(false).WithShowLabel(true).WithWidgetFrac(0.5f).WithDrawShadows(false), false), kNoTag, "vcontrols");



      pGraphics->AttachControl(new IVSlideSwitchControl(cell(0, 5).GetMidVPadded(buttonSize), sendDiff, "Only Diff",
        style.WithShowValue(false).WithShowLabel(true).WithWidgetFrac(0.5f).WithDrawShadows(false), false), kNoTag, "vcontrols");
      pGraphics->AttachControl(new IVKnobControl(cell(1, 5).GetMidVPadded(buttonSize), mix, "Mix", style, false), kNoTag, "vcontrols");

      //mid
      pGraphics->AttachControl(new IVSlideSwitchControl(cell(1, 1).Union(cell(1, 1)).GetMidVPadded(buttonSize), limiterType, "Limiter", style.WithWidgetFrac(0.5f).WithDrawShadows(false), true), kNoTag, "vcontrols");
      


      //down left
      pGraphics->AttachControl(new IVKnobControl(cell(2, 5).GetMidVPadded(buttonSize), outgain, "Gain Out", style, false), kNoTag, "vcontrols");
      pGraphics->AttachControl(new IVKnobControl(cell(0, 1).GetMidVPadded(buttonSize), dcR, "DC-R", style, false), kNoTag, "vcontrols");
      pGraphics->AttachControl(new IVKnobControl(cell(1, 3).GetMidVPadded(buttonSize), threshold, "Threshold", style, false), kNoTag, "vcontrols");

      pGraphics->AttachControl(new IVMeterControl<2>(cell(3, 5).Union(cell(4, 5)), "Output", style), kCtrlTagOutput);
      pGraphics->AttachControl(new IVMeterControl<2>(cell(0, 0).Union(cell(1, 0)), "Limit", style), kCtrlTagLim);

      pGraphics->AttachControl(new IVKnobControl(cell(1, 4).GetMidVPadded(buttonSize), maxTresh, "Max Thresh", style, false), kNoTag, "vcontrols");



      // Curve box
      pGraphics->AttachControl(new IVPlotControl(cell(2, 0).Union(cell(4, 4)), {
                                                              {COLOR_DARK_GRAY , [&](double x) { return getValueDisplay(x, this); } },
                                                              {COLOR_BLUE, [&](double x) { return getStatR(x, this); } },
                                                              {COLOR_RED, [&](double x) { return getStatL(x, this); } }


        }, 32, "IVPlotControl", style.WithShowLabel(false).WithDrawFrame(true)), kNoTag, "vcontrols");
      // Sliders
      int gfromr = 2;

      pGraphics->AttachControl(new IVSliderControl(sliderCell(2, 4, 2), cp4, "4", style2));
      pGraphics->AttachControl(new IVSliderControl(sliderCell(2, 3, 2), cp3, "3", style2));
      pGraphics->AttachControl(new IVSliderControl(sliderCell(2, 2, 2), cp2, "2", style2));
      pGraphics->AttachControl(new IVSliderControl(sliderCell(2, 1, 2), cp1, "1", style2));
      pGraphics->AttachControl(new IVSliderControl(sliderCell(2, 0, 2), cp0, "0", style2));

       

    // Left
   
  };


  // file2 << "started" << std::endl;
  // if(!mLatencyDelay)
    // file2 << "no latency delay" << std::endl;
  if(false)
    testPlug();
#endif
}

#if IPLUG_DSP


void IPlugEffect::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  const int nChans = NOutChansConnected();
  const double sampleRate = GetSampleRate();
  SetLatency(PLUG_LATENCY);

  double pan = GetParam(dcR)->Value();
  tresh = GetParam(threshold)->Value();
  mtreshp = GetParam(maxTresh)->Value();
  mtresh = tresh * mtreshp;
  mdcR = GetParam(dcR)->Value();
  sigmoid.setSteepness(GetParam(steep)->Value());
  double ogain = GetParam(outgain)->Value();
  int ltype = GetParam(limiterType)->Value();
  limiter.setType(Limiter::LType(ltype));

  mmix = GetParam(mix)->Value();
  bsendDiff = GetParam(sendDiff)->Value();
  breplot = GetParam(replot)->Value();

  bool update(false);

  for (int s = 0; s < 10; s++) {
    for (int i = 0; i < nChans; i++) {
      cur_stat[i][s] = 0.0;
    }
  }

  updateDisplay(_cp0, cp0, update);
  updateDisplay(_cp1, cp1, update);
  updateDisplay(_cp2, cp2, update);
  updateDisplay(_cp3, cp3, update);
  updateDisplay(_cp4, cp4, update);

  if(!UIClosed)
    if (atStartCount < 100) atStartCount++;
    else if(displayCount%1==0) GetUI()->SetAllControlsDirty();


  int id, id_prev;
  sample tmp_dc_diff[2] = { 0.0, 0.0 },
    delta[2] = { 0.0, 0.0 };
  double count[2] = { 0.0, 0.0 }, dt(0.0), lx(0.0), tmp, tmp_buff, s0, sp;

  if (nChans == 2) { // Stereo Signal
    // First process the new inputs buffer with dc block algo
    for (int s = 0; s < nFrames; s++) {
      id = (pro_pos + s + maxBuffSize) % maxBuffSize;
      id_prev = (pro_pos + s + maxBuffSize - 1) % maxBuffSize;
      for (int i = 0; i < nChans; i++) {
        

        input_cpy[i][id] = inputs[i][s];
        if (s == 0) {
          y_buffer[i][id] = inputs[i][s] - last_input[i] + mdcR * y_buffer[i][id_prev];
        }
        else {
          y_buffer[i][id] = inputs[i][s] - inputs[i][s - 1] + mdcR * y_buffer[i][id_prev];
        }


        // Send buffer to the mimiter
        process_buffer[i][id] = y_buffer[i][id];
        limiter.processSample(process_buffer[i][id], tresh*mtreshp);
      }
    }



    // Process the buffer for min/max algo
    for (int s = 0; s < nFrames; s++) {
      id = (pro_pos + s + maxBuffSize) % maxBuffSize;
      id_prev = (pro_pos + s + maxBuffSize - 1) % maxBuffSize;
      for (int i = 0; i < nChans; i++) {
        var[i] = process_buffer[i][id] - process_buffer[i][id_prev];
        out_buffer[i][id] = process_buffer[i][id];
        if (var[i] * lvar[i] <= 0.0) // On a step or sign change max/min dot
        {

          tmp_buff = getValue(process_buffer[i][id], i);
          

          dt = tmp_buff - process_buffer[i][id];
          tmp = (id + maxBuffSize - ldot[i]) % maxBuffSize;

          s0 = process_buffer[i][ldot[i]];
          sp = process_buffer[i][id];
          // si : process_buffer[i][cur_id]
          // u0 : last_buffer[i]
          // up : tmp_buff

          for (int t = 0; t < (id + maxBuffSize - ldot[i]) % maxBuffSize; t++) {
            int cur_id = (ldot[i] + t) % maxBuffSize;
            
            //lx = double(t) / tmp;
            if (breplot) {
              lx = sigmoid.get(double(t) / tmp);
              out_buffer[i][cur_id] = lx * tmp_buff + (1.0 - lx) * last_buffer[i];
            }
            
            else { // ui = u0 + a(i)(up - u0)
              lx = sigmoid.get((process_buffer[i][cur_id] - s0) / (sp - s0));
              out_buffer[i][cur_id] = last_buffer[i] + lx * (tmp_buff - last_buffer[i]);
            }

            if (bsendDiff) 
              out_buffer[i][cur_id] = out_buffer[i][cur_id] - process_buffer[i][cur_id];
            
          }

          last_delta[i] = dt;
          last_buffer[i] = tmp_buff;
          
          ldot[i] = id;
          ldot_age[i] = 0;
        }
        else // Same sign, do nothing
        {
          if (ldot_age[i] <= maxBuffSize)
            ldot_age[i]++;
        }

        lvar[i] = var[i];

        if (ldot_age[i] >= maxBuffSize) {
          ldot[i] = id;
        }
      }
    }

    double tmp = 0.0;
    // Normalize stat data
    for (int s = 0; s < 10; s++) {
      for (int i = 0; i < nChans; i++) {

        //if(count[i]>0.0)
        //  cur_stat[i][s] /= count[i];

        cur_stat[i][s] = cur_stat[i][s] / (1.0 + cur_stat[i][s]);
        prev_stat[i][s][displayCount] = cur_stat[i][s];

       my_stat[i][s] = 0.0;
        for (int d = 0; d < displayLoop; d++) {
          my_stat[i][s] += prev_stat[i][s][d];
        }
        my_stat[i][s] /= displayLoop;
      }
    }

    

    // Copy the buffer
    for (int s = 0; s < nFrames; s++) {
      id = (pro_pos + s + maxBuffSize - PLUG_LATENCY) % maxBuffSize;
      
      outputs[0][s] = ogain * (mmix * out_buffer[0][id] + (1.0 - mmix) * input_cpy[0][id]);
      outputs[1][s] = ogain * (mmix * out_buffer[1][id] + (1.0 - mmix) * input_cpy[1][id]);
    }

    last_input[0] = inputs[0][nFrames - 1];
    last_input[1] = inputs[1][nFrames - 1];

  } else { // Mono

  }

  pro_pos += nFrames;
  pro_pos %= maxBuffSize;


  mOutSender.ProcessBlock(outputs, nFrames, kCtrlTagOutput);
  //mLimSender.ProcessBlock(outputs, nFrames, kCtrlTagLim);

  isInit = true;
  ++displayCount %= displayLoop;
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
  SetLatency(PLUG_LATENCY);
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
  displayCount = 0;
  UIClosed = false;
}

/*
void IPlugEffect::OnActivate()
{
  Plugin::OnActivate(true);
  // file2 << "on activate" << std::endl;
}
*/
/*
void IPlugEffect::OnReset()
{
  Plugin::OnReset();
  this->SetLatency(PLUG_LATENCY);
  // file2 << "on reset 3 " << std::endl;
} */

sample IPlugEffect::getValue(const sample& s, const int &side)
{
  double x, ret;
  if (s > 0.0) x = s / tresh;
  else x = -s / tresh;

  if (x < .25) {
    if(x<.125)
    cur_stat[side][0] += 1.0;
    else
      cur_stat[side][1] += 1.0;
    ret = x * 4.0 * _cp0;
  }

  else if (x < .5) {
    if (x < .375)
      cur_stat[side][2] += 1.0;
    else
      cur_stat[side][3] += 1.0;
    ret = (x - .25) * 4.0 * (_cp1 - _cp0) + _cp0;
  }
  else if (x < .75) {
    if (x < .625)
      cur_stat[side][4] += 1.0;
    else
      cur_stat[side][5] += 1.0;
    ret = (x - .5) * 4.0 * (_cp2 - _cp1) + _cp1;
  }
  else if (x < 1.) {
    if (x < .875)
      cur_stat[side][6] += 1.0;
    else
      cur_stat[side][7] += 1.0;
    ret = (x - .75) * 4.0 * (_cp3 - _cp2) + _cp2;
  }
  else {
    if (x < (mtreshp -1.0) / 2.0 + 1.0)
      cur_stat[side][8] += 1.0;
    else
      cur_stat[side][9] += 1.0;

    ret = (x - 1.0) / (mtreshp - 1.0) * (_cp4 - _cp3) + _cp3;
  }

  if (s < 0.0)
    ret *= -1.0;

  return ret * tresh;
}

void IPlugEffect::testPlug()
{
  std::fstream file(test_file, std::ios::out | std::ios::trunc);

  const int size = PLUG_LATENCY / 2 - 3,
    nloop = 25,
    gate_s = 300,
    gate_e = 3500;
  sample** inputs, ** outputs;


  
  
  outputs = new sample * [2];
  inputs = new sample * [2];

  for (int i = 0; i < 2; i++){
    outputs[i] = new sample[size];
    inputs[i] = new sample[size];
  }

  for(int l=0;l<nloop;l++){
    for (int s = 0; s < size; s++) {
      for (int i = 0; i < 2; i++){
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


  for (int i = 0; i < 2; i++){
    delete[] outputs[i];
    delete[] inputs[i];
  }
  delete[] outputs;
  delete[] inputs;

  file.close();
  system("pause");
}
void IPlugEffect::updateCurve()
{
}
#endif

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
  else out = 0.0;
  return out;
}

void Limiter::setChannels(const int& n)
{
  nChans = n;
  diff = new sample*[nChans];
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

Sigmoid::Sigmoid() : steep(0.0), c(1.0), v(1.0), type(SIG)
{
  setSteepness(1.0);
}

Sigmoid::~Sigmoid()
{
}

double Sigmoid::get(const double& x) const
{
  switch (type)
  {
  case Sigmoid::SIG:
    return sig(x);
    break;
  case Sigmoid::LIN:
    return linear(x);
    break;
  case Sigmoid::REV:
    return rev(x);
    break;
  default:
    return linear(x);
    break;
  }
}

double Sigmoid::rev(const double& x) const
{
  return -log(c / (v + x) - 1.0) / steep + 0.5;
}

double Sigmoid::sig(const double& x) const
{
  return c / (1.0 + exp(-steep * (x - 0.5))) - v;
}

double Sigmoid::linear(const double& x) const
{
  return x;
}

void Sigmoid::setSteepness(const double& steepness)
{
  if (steep == steepness)
    return;
  if (steepness < 0.0) {
    type = SType::REV;
    steep = -steepness;
    v = (1.0 + exp(-steep / 2.0)) / (exp(steep / 2) - exp(-steep / 2.0));
    c = v * (1.0 + exp(steep / 2.0));
  }
  else if (steepness > 0.0) {
    type = SType::SIG;
    steep = steepness;
    v = (1.0 + exp(-steep / 2.0)) / (exp(steep / 2) - exp(-steep / 2.0));
    c = v * (1.0 + exp(steep / 2.0));
  }
  else {
    type = SType::LIN;
  }
}
