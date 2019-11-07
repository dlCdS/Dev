// Dogodo.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//


#include "Main.h"
#include <Weather.h>


int main(int argc, char* argv[]) {
	SetLogLevel(LOG_LEVEL::LERROR);

	bool *done = new bool;
	MainEventSolver::Associate<KeyPressedEvent>(new SolveKeypressed());
	MainEventSolver::Associate<ExitEvent>(new SolveExitEvent(done));

	Main::Init();
	GE::Init();

	GE::SetWindowSize();
	GE::Start();

	Main::Load();

	Main::Build();
	ContainerWidget *maindisplay = new ContainerWidget(square_d(0, 0, 1, 1));
	maindisplay->addWidget(Main::GetContainer());
	
	/*
	AnimationEditor animEditor;
	animEditor.build();
	maindisplay->addWidget(animEditor.getContainer());

	
	GameDisplay::setDisplaySize(ge_pi(500, 500));
	GameDisplay::setTileSize(20);
	GameDisplay::Build();
	maindisplay->addWidget(GameDisplay::GetContainer());
	*/

	GE::AddWidgetToWindow(maindisplay);
	//GE::createWindowWidget();
	
							 
	//GE::SaveWindowConfiguration("WindowConfiguration.xml.test");
	
	/*
	GE::ResetWindow();
	GE::LoadWindowConfiguration("WindowConfiguration.xml.test");*/


	*done = false;
	while (!(*done)) {
		GE::Step();
		MainEventSolver::SolveSdlEvent();
		MainEventSolver::SolveMonitoringEvent();
	}

	GE::Quit();
	return 0;
}