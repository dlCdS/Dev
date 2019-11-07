// FunctionDrawer.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <iostream>

#include <MainEventSolver.h>
#include "CurtilModel.h"
#include <FractalModel.h>
#include <CudaFreqDrawer.h>
#include <SoundHandler.h>
#include <SdlSound.h>

bool done = false;

ContainerWidget* maindisplay;

// SDL Configuration

const ge_i frame_duration = 40;
const ge_i frame_rate = 1;
const ge_i size = 200;
const ge_pi wsize = { 1920, 1000 };
const ge_i factor = 1;
const ge_d ratio = 0.5;
const ge_d rotate = -45.0;
const ge_pd center = { -0.0, 1.0 };

const bool play_sound = false;

// #define FRACTAL

bool step();

void initGraphicalEngine() {
	SetLogLevel(LOG_LEVEL::LERROR);

	GE::Init();
	GE::SetWindowSize({ wsize.w, wsize.h });
	GE::Start(frame_duration);

	MainEventSolver::Associate<KeyPressedEvent>(new SolveKeypressed());
	MainEventSolver::Associate<ExitEvent>(new SolveExitEvent(&done));

	maindisplay = new ContainerWidget(square_d(0, 0, 1, 1));
}

bool cudaFreqLoop();

void GraphicalLoop() {
   	GE::AddWidgetToWindow(maindisplay);
	ge_i frame = 0;
	while (!done) {
		Clocks::start("main loop");
		Clocks::start("tested_model");
#ifdef FRACTAL
		done = step();
#else
		done = cudaFreqLoop();
#endif
		Clocks::stop("tested_model");
		if ((frame++) % (frame_rate) == 0 && !done) {
			Clocks::start("GE::Step()");
			if(play_sound)
				SdlSound::PlayAudio();
			GE::Step();
			Clocks::stop("GE::Step()");
		}
		MainEventSolver::SolveSdlEvent();
		MainEventSolver::SolveMonitoringEvent();
		Clocks::stop("main loop");
	}
	if(play_sound)
		SdlSound::WaitAudio();
	SdlSound::FreeWAV();
	GE::Quit();
}

CurtilModel model;
FractalModel fractal;
CudaFractalModel sdlInterface;
CudaFreqDrawer freqDrawer;
SoundHandler sp;

void cudaFractalLoop();

bool step() {
	cudaFractalLoop();
	return false;
}


void curtisModel() {

	model.setSize({ wsize.w, wsize.h }, factor);

	GridWidget* gridWidget = new GridWidget({ 2, 2 });
	SetColourWidget* scw = new SetColourWidget();
	SetColourWidget* scu = new SetColourWidget();
	SetColourWidget* scv = new SetColourWidget();
	SetColourWidget* scp = new SetColourWidget();
	gridWidget->addWidget(scu);
	gridWidget->addWidget(scv);
	gridWidget->addWidget(scp);
	gridWidget->addWidget(scw);

	maindisplay->addWidget(gridWidget);

	model.setColourWidget(scw);
	model.setUVPWidget(scu, scv, scp);

	model.generate(0.001, 0.04, 0.01, 0.1);
}

void spModel() {
	sp.printDevices();
	sp.setFile("E:\\Programmes\\VS2017\\CdS_Data\\FunctionDrawer\\Data\\sine\\sine300to500LtoR.wav");
	sp.configure(4096, 1.0, frame_duration);
	sp.enablePanComputation(true);
	sp.testApi();
	Clocks::report();
}

void fractalModel() {
	fractal.setSize({ wsize.w, wsize.h }, factor);

	SetColourWidget* scw = new SetColourWidget();

	maindisplay->addWidget(scw);

	fractal.setColourWidget(scw);

	fractal.generate({ -0.5, 0.0 }, { ratio, wsize.h*ratio/ wsize.w });
}

void cudaFractalModel() {
	sdlInterface.setSize({ wsize.w, wsize.h }, factor);

	SetColourWidget* scw = new SetColourWidget();

	maindisplay->addWidget(scw);

	sdlInterface.setColourWidget(scw);

	sdlInterface.generate(center, rotate, { ratio, wsize.h*ratio / wsize.w });

}

void cudaFractalLoop() {
	Clocks::start("cudaFractalLoop");
	sdlInterface.cycle();
	Clocks::stop("cudaFractalLoop");
}

void cudaFreqDrawer() {
	freqDrawer.setSize({ wsize.w, wsize.h }, factor);
	const int sample_size = 4096*2;
	// sine300to500LtoR
   //  sine10to2kLtoR
   // sound\\overkill_190303.wav
	std::string filename = "E:\\Programmes\\VS2017\\CdS_Data\\FunctionDrawer\\Data\\sound\\Ferme.wav";
	//std::string filename = "E:\\Programmes\\VS2017\\CdS_Data\\FunctionDrawer\\Data\\sine\\sine10to2kLtoR.wav";

	SetColourWidget* scw = new SetColourWidget();

	maindisplay->addWidget(scw);

	freqDrawer.setColourWidget(scw);

	freqDrawer.generate(center, rotate, { ratio, wsize.h * ratio / wsize.w }, { 1.0, 1.0 }, 5.0);
	// sp.setFile("E:\\Programmes\\VS2017\\FunctionDrawer\\Data\\sine\\sine10to2kLtoR.wav", 48000.0);
	sp.setFile(filename, 48000.0);
	sp.configure(sample_size, 1.0, frame_duration);
	sp.enablePanComputation(true);
	ge_d min_freq = sp.getFFTindex(200.0);
	ge_d max_freq = sp.getFFTindex(2500.0);
	
	freqDrawer.setFrequencyIndexInterval(min_freq, max_freq, sample_size);

	SdlSound::LoadWAV(filename);
}

bool cudaFreqLoop() {
	
	Clocks::start("getTicks");
	while (sp.nextTick()) {
		;;
	}
	Clocks::stop("getTicks");


	Clocks::start("consume");
	freqDrawer.consume(sp.consumme());
	Clocks::stop("consume");
	Clocks::start("process");
	freqDrawer.process();
	Clocks::stop("process");
	Clocks::start("draw");
	freqDrawer.draw();
	Clocks::stop("draw");


	return sp.noData();
}

 // #define TEST_SP_MODEL

int main(int argc, char* argv[]) {
		
	initGraphicalEngine();

#ifdef FRACTAL
	cudaFractalModel();
#else
	cudaFreqDrawer();
#endif

	GraphicalLoop();

	Clocks::report();

	return 0;
	
}