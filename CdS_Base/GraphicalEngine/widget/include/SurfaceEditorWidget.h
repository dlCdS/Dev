#pragma once
#include "GetColourWidget.h"

class SurfaceEditorWidget :
	public GetColorWidget
{
public:
	SurfaceEditorWidget();
	virtual ~SurfaceEditorWidget();

protected:
	virtual void draw(SDL_Renderer *renderer);
	virtual void postComputeRelative(const ge_pi &pen);
};

