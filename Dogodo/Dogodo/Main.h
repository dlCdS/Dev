#pragma once
#include <Widgetable.h>

#include <iostream>
#include <stdio.h>
#include <MainEventSolver.h>
#include "GameDisplay.h"
#include "ContentProvider.h"

#define CONFIG_INIT_FILENAME "Config.ini"

class ConfigInit : public XML::Compliant {
public:
	ConfigInit();
	virtual std::string XMLName() const;

public:
	virtual void associate();
	std::string _workspace;
};

class Main : public Widgetable
{
public:
	~Main();

	static void Init();
	static void Build();
	static void Load();
	static Widget *GetContainer();

protected:

	void selectWorkspace(void *v);
	void saveWorkspace(void *v);
	std::string getWorkspace();
	void setWorkspace(std::string *str);

	std::string XMLName() const;
	virtual void associate();

private:

	std::string _workDir;
	Main();
	static Main _singleton;
	static ConfigInit _configInit;

};

