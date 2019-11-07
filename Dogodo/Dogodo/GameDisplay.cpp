#include "GameDisplay.h"

GameDisplay GameDisplay::_singleton = GameDisplay() ;

std::string GameDisplay::XMLName() const
{
	return staticXMLName();
}

std::string GameDisplay::staticXMLName()
{
	return"GameDisplay";
}

void GameDisplay::Build()
{
	Clocks::addClock("GameDisplay::Build");
	Clocks::start("GameDisplay::Build");
	_singleton.build();
	Log(LINFO, "Built Game Diplay in ", Clocks::stop("GameDisplay::Build"));
}

Widget * GameDisplay::GetContainer()
{
	return _singleton.getContainer();
}

void GameDisplay::setDisplaySize(const ge_pi & dim)
{
	_singleton._dim = dim;
}

void GameDisplay::setTileSize(const ge_i & size)
{
	_singleton._tileSize = size;
}

GameDisplay::GameDisplay() : Widgetable("Game_Display")
{
}

void GameDisplay::associate()
{
	MoveGridWidget *mg = new MoveGridWidget(_dim);
	mg->setRelativeProportion(square_d(-1, -1, 0.8, 0.8));
	mg->setGridAbstractSize(20);
	mg->setAbstractPos(0, 0);
	WAssociateWidget("Grid", mg);
	for (int i = 0; i < mg->getGridSize(); i++) {
			
		/*
		ColoredSquareWidget *csw = new ColoredSquareWidget();
		csw->setSize(ge_pi(20, 20));
		csw->setColor(i * 5, i * 7, i * 11);
		mg->addWidget(csw);
		

		*/
		
		
		IconWidget *icw = new IconWidget(NULL, AnimationDataBase::requestAnimation(DEFAULTANIMATION));
		icw->setSize(ge_pi(20, 20));
		mg->addWidget(icw);
		
	}
	mg->scalingReduction();
}


GameDisplay::~GameDisplay()
{
}
