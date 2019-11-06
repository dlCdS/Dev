#pragma once
#include <Common.h>
#include <SetColourWidget.h>


#define e(f, i, j) (f[i][j])
#define loop() for(ge_i i=0; i<_size.w; i++) for(ge_i j=0; j<_size.h; j++)

class VirtualModel
{
public:
	VirtualModel();
	virtual ~VirtualModel();

	virtual void setColourWidget(SetColourWidget* scw);

	void setSize(const ge_pi& size, const ge_i& zoom);

	void cycle();


protected:

	virtual void draw() = 0;

	virtual void mainLoop() = 0;

	void initWidget(SetColourWidget* widget);

	SetColourWidget* _scw;
	ge_pi _size;
	ge_i _zoom;
};

