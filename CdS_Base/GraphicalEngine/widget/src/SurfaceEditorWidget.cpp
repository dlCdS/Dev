#include "SurfaceEditorWidget.h"



SurfaceEditorWidget::SurfaceEditorWidget() : 
	GetColorWidget(),
	Widget(NULL, false, true, true, true, false, square_d(-1, -1, -1, -1))
{
}


SurfaceEditorWidget::~SurfaceEditorWidget()
{
}

void SurfaceEditorWidget::draw(SDL_Renderer * renderer)
{
	GetColorWidget::draw(renderer);
}

void SurfaceEditorWidget::postComputeRelative(const ge_pi & pen)
{
	SDL_Rect pr = { 0, 0, 0, 0 };
	getFirstNotEmptyRelRect(pr);
	_size = { pr.w - pen.w , pr.h - pen.h };
	ge_d factor;
	ge_pd relside;
	if (_surface != NULL){
		relside = { (ge_d)_size.w / _surface->w, (ge_d)_size.h / _surface->h };
		factor = min(relside.w, relside.h);
		_size = { (ge_i)(factor * _surface->w), (ge_i)(factor * _surface->h) };
	}
	if (_sqd.dim.w < 0.0)
		_rel.w = _size.w;
	if (_sqd.dim.h < 0.0)
		_rel.h = _size.h;
}


