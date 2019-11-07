#pragma once
#include "InstanceModel.h"

class Tile
{
public:
	Tile();
	~Tile();

protected:

};


class TileModel : public Tile, public InstanceModel
{
public:
	TileModel();
	~TileModel();
};
