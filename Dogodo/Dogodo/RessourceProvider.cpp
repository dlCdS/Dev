#include "RessourceProvider.h"


RessourceProvider RessourceProvider::_sing = RessourceProvider();

void RessourceProvider::addRessource(void * v)
{
	loadRessource(NULL);
	buildList();
	Log(LINFO, "Ressource added");
}

void RessourceProvider::removeRessource(void * v)
{
	int id = Common::Cast<int>(*(std::string*)v);
	if (id >= 0 && id < _data.size()) {
		delete _data[id];
		for (int i = id; i < _data.size()-1; i++) {
			_data[i] = _data[i + 1];
		}
		_data.pop_back();
	}
	buildList();
}

void RessourceProvider::saveRessource(void * v)
{
	save(RESSOURCE_PROVIDER_FILE);
}

void RessourceProvider::iconClicked(void * v)
{
	Log(LINFO, "Icon Clicked callback");
	AnimationEditor::SetAnimation((Animation*)v);
	if (!_container.setPage(ANIMATION_EDITOR_PAGE_NAME)) {
		_container.addWidget(AnimationEditor::GetContainer(), ANIMATION_EDITOR_PAGE_NAME);
		_container.setPage(ANIMATION_EDITOR_PAGE_NAME);
		AnimationEditor::GetSingleton()->setAsPage(this);
	}
	else {
		AnimationEditor::GetSingleton()->setAsPage(this);
	}
}

void RessourceProvider::buildList()
{
	_ressource.freeAll();
	int count(0);
	for (auto d : _data) {
		ContainerWidget *cont = new ContainerWidget();
		ClickWidget *cw = new ClickWidget();
		cw->setText("Delete", Common::Cast(count++));
		cw->useTextAsParameter();
		cw->setCallback(VoidedCallbackFunction(RessourceProvider, removeRessource), true);
		NamedIconWidget *niw = new NamedIconWidget();
		niw->setTextSize(LetterLib::NORMAL);
		niw->associateIcon(d->_anim, new XML::String(&d->_name));
		niw->setIconClicked(VoidedCallbackFunction(RessourceProvider, iconClicked));
		cont->addWidget(cw);
		cont->addWidget(niw);
		_ressource.addWidget(cont);
	}
	_ressource.sqdHasChanged();
}

std::string RessourceProvider::XMLName() const
{
	return "RessourceProvider";
}

XML::Parsable * RessourceProvider::loadRessource(void * v)
{
	_data.push_back(new RessourceData("NoName", AnimationDataBase::getDefaultAnimation()));
	return _data[_data.size() - 1];
}

void RessourceProvider::associate()
{
	WAssociateWidget("Up", new ContainerWidget());
	WAssociateWidget("Down", new ContainerWidget());
	WAssociateWidget("Down:Ressources", &_ressource);
	XMLAssociateSubBeacon("ressources", _data.begin(), _data.end(), SubBeaconLoadFunction(RessourceProvider, loadRessource));

	WAddCallback("Add", VoidedCallbackFunction(RessourceProvider, addRessource));
	WAssociateBeacon("Add", "Up");
	WAddCallback("Save", VoidedCallbackFunction(RessourceProvider, saveRessource));
	WAssociateBeacon("Save", "Up");
	_sing.buildList();
}

RessourceProvider::RessourceProvider() : Widgetable("Ressource_Provider", true)
{
	_ressource.setShared();
}


RessourceProvider::~RessourceProvider()
{
}

void RessourceProvider::Build()
{
	_sing.build();
}

Widget * RessourceProvider::GetContainer()
{
	return _sing.getContainer();
}

Widgetable * RessourceProvider::GetSingleton()
{
	return &_sing;
}

void RessourceProvider::Load()
{
	_sing.load(RESSOURCE_PROVIDER_FILE);
}

RessourceProvider::RessourceData::RessourceData() :
	_name("RessourceData"),
	_anim(NULL)
{
}

RessourceProvider::RessourceData::RessourceData(const std::string & name, Animation * anim) :
	XML::Parsable(),
	_name(name),
	_anim(anim)
{
}

std::string RessourceProvider::RessourceData::XMLName() const
{
	return staticXMLName();
}

std::string RessourceProvider::RessourceData::staticXMLName()
{
	return "RessourceData";
}

void RessourceProvider::RessourceData::associate()
{
	XMLAssociateField("name", new XML::String(&_name));
	XMLAddReference("animation", 
		SubBeaconGetReference(Animation, getFilename, _anim), 
		SubBeaconLoadFunction(RessourceData, loadAnimation));
}

void RessourceProvider::RessourceData::loadAnimation(std::string * str)
{
	_anim = AnimationDataBase::requestAnimation(*str);
}
