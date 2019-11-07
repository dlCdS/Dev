#include "ContentProvider.h"


ContentProvider ContentProvider::_sing = ContentProvider();

ContentProvider::ContentProvider() : Widgetable("Content_Provider", true)
{
}

std::string ContentProvider::XMLName() const
{
	return "ContentProvider";
}

void ContentProvider::associate()
{

	WAssociateWidget("Left", new ContainerWidget(square_d(-1, -1, 0.3, 1)));

	WAssociateWidget("Left:Up", new ContainerWidget(square_d(-1, -1, 1, 0.2)));
	WAssociateWidget("Left:Up:Ressource", new ListWidget());
	WAssociateWidget("Left:Up:Ressource:Title", new ContainerWidget());
	WAssociateWidget("Left:Up:Ressource:Display", RessourceProvider::GetContainer());

	WAssociateWidget("Left:Down", new ContainerWidget(square_d(-1, 0.2, 1, 0.8)));

	WAssociateWidget("Right", new ContainerWidget(square_d(0.3, 0, 0.7, 1)));

	WAddPage("Ressources", "Left:Up:Ressource:Title", RessourceProvider::GetSingleton());
}


ContentProvider::~ContentProvider()
{
}

void ContentProvider::Init()
{
	_sing._container.freeAll();
	_sing.load(CONTENT_PROVIDER_FILENAME);
}

void ContentProvider::Build()
{
	_sing.build();
}

Widget * ContentProvider::GetContainer()
{
	return _sing.getContainer();
}

Widgetable * ContentProvider::GetSingleton()
{
	return &_sing;
}
