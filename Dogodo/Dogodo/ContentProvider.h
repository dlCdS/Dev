#pragma once
#include <Widgetable.h>
#include "RessourceProvider.h"

#define CONTENT_PROVIDER_FILENAME "ContentProvider.xml"
class ContentProvider :
	public Widgetable
{
public:
	virtual ~ContentProvider();

	static void Init();
	static void Build();
	static Widget *GetContainer();
	static Widgetable *GetSingleton();

protected:

	ContentProvider();
	std::string XMLName() const;
	virtual void associate();

	static ContentProvider _sing;
};

