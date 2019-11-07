#pragma once
#include <Widgetable.h>
#include <Ressource.h>
#include "AnimationEditor.h"

#define RESSOURCE_PROVIDER_FILE "Ressource.xml"

class RessourceProvider :
	public Widgetable
{
public:
	class RessourceData : public XML::Parsable {
	public:
		RessourceData();
		RessourceData(const std::string &name, Animation *anim);
		std::string _name;
		Animation *_anim;


		virtual std::string XMLName() const;
		static std::string staticXMLName();

		virtual void associate();
		virtual void loadAnimation(std::string *str);
	};
public:
	~RessourceProvider();

	static void Build();
	static Widget *GetContainer();
	static Widgetable *GetSingleton();
	static void Load();

private:
	void addRessource(void *v);
	void removeRessource(void *v);
	void saveRessource(void *v);
	void iconClicked(void *v);

	void buildList();

	std::string XMLName() const;

	XML::Parsable *loadRessource(void *v);
	virtual void associate();

	RessourceProvider();

	ListWidget _ressource;
	std::vector<RessourceData*> _data;
	static RessourceProvider _sing;
};

