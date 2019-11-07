#pragma once
#include <Widgetable.h>
#include <Clocks.h>
class GameDisplay :
	public Widgetable
{
public:
	virtual ~GameDisplay();


	virtual std::string XMLName() const;
	static std::string staticXMLName();

	static void Build();
	static Widget *GetContainer();

	static void setDisplaySize(const ge_pi &dim);
	static void setTileSize(const ge_i &size);

private:
	GameDisplay();

	virtual void associate();
	ge_pi _dim;
	ge_i _tileSize;

	static GameDisplay _singleton;

};

