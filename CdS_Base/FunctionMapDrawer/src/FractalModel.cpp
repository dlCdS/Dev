#include "FractalModel.h"

FractalModel::FractalModel()
{
}

FractalModel::~FractalModel()
{
	Common::Deallocate(_size.w, _size.h, _c);
	Common::Deallocate(_size.w, _size.h, _cur);
	Common::Deallocate(_size.w, _size.h, _transform);
}

struct thread_data {
	SetColourWidget* scw;
	FractalModel* model;
	std::complex<ge_d>** source;
	ge_i fi, ti, fj, tj;
};

static int parallelDraw(void* ptr) {
	thread_data *str = (thread_data*)(ptr);
	for(int i=str->fi; i< str->ti; i++)
		for (int j = str->fj; j < str->tj; j++) {
			str->scw->setPixelColor(i, j, str->model->getRatioNormColor(i, j, str->source));
		}
	return 0;
}

void FractalModel::draw()
{
	const int nthread(2);
	const bool use_thread(true);
	static ge_i frame = 0;
	Clocks::start("fdraw");
	SDL_Rect rect;
	rect.h = _zoom; rect.w = _zoom;
	if(!use_thread){
		if (frame++ % 1 == 0) {
			loop() {
				rect.x = _zoom * i; rect.y = _zoom * j;
				Clocks::start("fdrawloop");
				_scw->setPixelColor(i, j, getRatioNormColor(i, j, _cur));
				// _scw->setRectColor(rect, getRatioNormColor(i, j, _cur));
				Clocks::stop("fdrawloop");

			}
		}
	}
	else {
		SDL_Thread *(thread[nthread][nthread]);
		struct thread_data td[nthread][nthread];
		for(int i=0; i< nthread;i++)
			for (int j = 0; j < nthread; j++) {
				td[i][j].model = this;
				td[i][j].source = _cur;
				td[i][j].scw = _scw;
				td[i][j].fi = i* _size.w/nthread;
				td[i][j].ti = (i+1) * _size.w / nthread;
				td[i][j].fj = j * _size.h / nthread;
				td[i][j].tj = (j + 1) * _size.h / nthread;
				thread[i][j] = SDL_CreateThread(parallelDraw, "parallelDraw", &td[i][j]);
			}
		int ret;
		
		for (int i = 0; i < nthread; i++)
			for (int j = 0; j < nthread; j++)
				SDL_WaitThread(thread[i][j], &ret);
	}
	Clocks::stop("fdraw");
}

void FractalModel::mainLoop()
{
	Clocks::start("FractalModelmainLoop");
	loop() {
		e(_cur, i, j) = e(_cur, i, j) * e(_cur, i, j) + e(_c, i, j);
		toreSpace(i, j, _cur);
	}
	Clocks::stop("FractalModelmainLoop");
}

void FractalModel::toreSpace(const ge_i& i, const ge_i& j, std::complex<ge_d>** source)
{
	ge_d val;
	if (e(source, i, j).real() < -1 * _tore.w){
		val = e(source, i, j).real() + 2.0 * _tore.w * int(-1.0 * e(source, i, j).real() / _tore.w / 2.0 + 2.0);
		e(source, i, j).real(val);
	}
	else if (e(source, i, j).real() >= _tore.w){
		val = e(source, i, j).real() - (int(e(source, i, j).real() / _tore.w / 2.0) * 2.0 * _tore.w);
		e(source, i, j).real(val);
	}

	if (e(source, i, j).imag() < -1 * _tore.h) {
		val = e(source, i, j).imag() + 2.0 * _tore.h * int(-1.0 * e(source, i, j).imag() / _tore.h / 2.0 + 2.0);
		e(source, i, j).imag(val);
	}
	else if (e(source, i, j).imag() >= _tore.h) {
		val = e(source, i, j).imag() - (int(e(source, i, j).imag() / _tore.h / 2.0) * 2.0 * _tore.h);
		e(source, i, j).imag(val);
	}
}

Color FractalModel::getRatioNormColor(const ge_i& i, const ge_i& j, std::complex<ge_d>** source)
{

	Clocks::start("getRatioNormColor");
	ge_d r(0.0), g(0.0), b(0.0);
	b =e(source, i, j).real() * e(source, i, j).real() + e(source, i, j).imag() * e(source, i, j).imag();
	b = sqrt(b);
	r = (1.0 + e(source, i, j).real() / b) * 255.0 / 2.0;
	g = (1.0 + e(source, i, j).imag() / b) * 255.0 / 2.0;
	b = std::arg(e(source, i, j)) * 255.0 / M_PI;
	
	Clocks::stop("getRatioNormColor");
	return Color(r, g, b);
}

void FractalModel::publish()
{
	_scw->updateTexture();
}

void FractalModel::generate(const ge_pd& center, const ge_pd& contraction, const ge_pd& tore)
{
	_c = Common::Allocate<std::complex<ge_d>>(_size.w, _size.h);
	_cur = Common::Allocate<std::complex<ge_d>>(_size.w, _size.h);
	_transform = Common::Allocate<std::complex<ge_d>>(_size.w, _size.h);
	_center = center;
	_contraction = contraction;
	_tore = tore;
	loop() {
		e(_transform, i, j) = { 0.0 };

		e(_c, i, j) = { ge_d(ge_d(i - _size.w / 2) * 2.0 * _contraction.w / ge_d(_size.w) + _center.w),
						ge_d(ge_d(j - _size.h / 2) * 2.0 * _contraction.h / ge_d(_size.h) + _center.h) };
		e(_cur, i, j) = e(_c, i, j);
		if ((i == 0 && (j == _size.h - 1 || j == 0)) || (i == _size.w - 1 && (j == _size.h - 1 || j == 0)))
			Log(LINFO, i, " ", j, " ", e(_c, i, j));
	}
}
