#include "Main.h"

Main Main::_singleton = Main();
ConfigInit Main::_configInit = ConfigInit();

Main::Main() : Widgetable("Main", true),
_workDir("No Workspace defined...")
{
}


Main::~Main()
{
}

void Main::Init()
{
	_configInit.load(CONFIG_INIT_FILENAME);
	_singleton._workDir = _configInit._workspace;
	_singleton.setWorkspace(&_singleton._workDir);
}

void Main::Build()
{
	_singleton.build();
}

void Main::Load()
{
	RessourceProvider::Load();
}

Widget * Main::GetContainer()
{
	return _singleton.getContainer();
}

std::string Main::getWorkspace()
{
	return _workDir;
}

void Main::setWorkspace(std::string * str)
{
	Workspace::SetPath(*str);
}

std::string Main::XMLName() const
{
	return "Main";
}

void Main::associate()
{
	WAssociateWidget("Browseandsave", new ContainerWidget());
	XMLAssociateField("Workspace", new XML::String(&_workDir));

	WAddCallback("Browse_Workspace", VoidedCallbackFunction(Main, selectWorkspace));
	WAddCallback("Save", VoidedCallbackFunction(Main, saveWorkspace));

	
	WAssociateBeacon("Browse_Workspace", "Browseandsave");
	WAssociateBeacon("Save", "Browseandsave");
	WAssociateField("Workspace", "", new XML::String(&_workDir));


	WAddPage("Animation_Editor", "", AnimationEditor::GetSingleton());
	WAddPage("Content_Provider", "", ContentProvider::GetSingleton());
}

void Main::selectWorkspace(void *v)
{
	Workspace::SetPath(Workspace::Explore());
	_workDir = Workspace::Path();
	variableChanged(&_workDir);
}

void Main::saveWorkspace(void * v)
{
	_configInit._workspace = _workDir;
	Workspace::SetPath(Workspace::StartPath());
	_configInit.save(CONFIG_INIT_FILENAME);
	Workspace::SetPath(_workDir);
	std::fstream f("Save.here", std::ios::trunc | std::ios::out);
	save("Main.xml");


}

ConfigInit::ConfigInit() : XML::Compliant("ConfigInit")
{
}

std::string ConfigInit::XMLName() const
{
	return "ConfigInit";
}

void ConfigInit::associate()
{
	XMLAssociateField("Workspace", new XML::String(&_workspace));
}
