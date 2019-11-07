#include "CurtilModel.h"


CurtilModel::CurtilModel() : VirtualModel(), _p(NULL), _u(NULL), _v(NULL)
{
}


CurtilModel::~CurtilModel()
{
	deleteModel();
}


void CurtilModel::setUVPWidget(SetColourWidget* scu, SetColourWidget* scv, SetColourWidget* scp)
{
	_scu = scu; _scv = scv; _scp = scp;
	initWidget(_scu); initWidget(_scv); initWidget(_scp);
}


void CurtilModel::generate(const ge_d &mu, const ge_d &kappa,
	const ge_d &ksi, const ge_d &tau)
{
	srand(time(NULL));  
	_mu = mu; _kappa = kappa; _ksi = ksi; _tau = tau;

	_p = Common::Allocate<ge_d>(_size.w, _size.h);
	_h = Common::Allocate<ge_d>(_size.w, _size.h);

	_u = Common::Allocate<ge_d>(_size.w, _size.h);
	_v = Common::Allocate<ge_d>(_size.w, _size.h);

	_up = Common::Allocate<ge_d>(_size.w, _size.h);
	_vp = Common::Allocate<ge_d>(_size.w, _size.h);

	_nhu = Common::Allocate<ge_d>(_size.w, _size.h);
	_nhv = Common::Allocate<ge_d>(_size.w, _size.h);

	_c = Common::Allocate<ge_d>(_size.w, _size.h);

	_wet = Common::Allocate<ge_pi*>(_size.w, _size.h);

	initialize();
	generateHeight();
}

void CurtilModel::initialize()
{
	loop() {
		e(_p, i, j) = 2.0;
		e(_h, i, j) = 0.0;
		_u[i][j] = 0.0;
		_v[i][j] = 0.0;
		_nhu[i][j] = 0.0;
		_nhv[i][j] = 0.0;
		_c[i][j] = 0.0;
		_wet[i][j] = NULL;
	}
	enforceBoundaryCondition(_u, _v);
}

void CurtilModel::mainLoop()
{
	moveWater();
	transfertPigment();
	transferPigment();
	simulateCapilarityFlow();
}

void CurtilModel::moveWater()
{
	Clocks::start("updateVelocities");
	updateVelocities();
	Clocks::stop("updateVelocities");
	Clocks::start("relaxDivergence");
	relaxDivergence();
	Clocks::stop("relaxDivergence");
}

void CurtilModel::enforceBoundaryCondition(ge_d **u, ge_d **v)
{
	for (int i = 0; i < _size.w; i++) {
		e(u, i, 0) = 0.0;
		e(u, i, _size.h - 1) = 0.0;
		e(v, i, 0) = 0.0;
		e(v, i, _size.h - 1) = 0.0;
		e(v, i, _size.h - 2) = 0.0;
	}
	for (int i = 0; i < _size.h; i++) {
		e(u, 0, i) = 0.0;
		e(u, _size.w - 1, i) = 0.0;
		e(u, _size.w - 2, i) = 0.0;
		e(v, 0, i) = 0.0;
		e(v, _size.w - 1, i) = 0.0;
	}
}

void CurtilModel::relaxDivergence()
{
	static bool zero_occured(false);
	ge_d t(0), delta(_tau), delta_max(_tau);
	ge_i count(0), max_iteration(100);
	bool spy(false);
		
	loop() {
		e(_up, i, j) = e(_u, i, j);
		e(_vp, i, j) = e(_v, i, j);
	}
	while(delta_max>=_tau && count++ < max_iteration){
		delta_max = 0.0;
		safeloop(1) {

			if (i == 0 && j == 1)
				spy = true;
			delta = 0.0;

				delta = e(_u, i, j) - e(_u, i - 1, j);

				delta += e(_v, i, j) - e(_v, i, j - 1);
			delta *= _ksi;
			
			if (delta != 0.0) {
				e(_p, i, j) -= delta;
				if(e(_p, i, j) < 0.0)
					e(_p, i, j) = 0.0;
				else {
					e(_c, i, j) -= delta;
					ifudiff(i+2) {
						e(_up, i, j) -= delta;
						
					}
					if (i > 1) e(_up, i - 1, j) += delta;
					ifvdiff(j+2) {
						e(_vp, i, j) -= delta;
						
					}
					if (j > 1) e(_vp, i, j - 1) += delta;
					delta_max = max(abs(delta), delta_max);
				}
			}

			if ((j == 7 || j == 6) && i == 0) {
				Log(LINFO, i, " : ", j, " u ",
					e(_up, i, j), " v ",
					e(_vp, i, j), " u ",
					e(_up, i - 1, j), " v ",
					e(_vp, i, j - 1), " delta ",
					delta);
				spy = false;
				zero_occured = true;
			}
		}

		if (count >= max_iteration)
			Log(LINFO, "Relax divergence exited because over iteration limit, delta_max is ", delta_max);
		loop() {
			e(_u, i, j) = e(_up, i, j);
			e(_v, i, j) = e(_vp, i, j);
		}
	}
	
}

void checkEquality(ge_d** v, const int &size) {
	ge_d last, diff;
	bool found(false);
	for (int j = 1; j < size-1; j++) {
		last = v[1][j];
		for (int i = 2; i < size-1; i++) {
			diff = v[i][j] - v[i - 1][j];
			if (diff != 0.0) {
				found = true;
				Log(LINFO, i, ",", j, " ", diff);
			}
		}
	}
	if(found) {
		Log(LINFO, "Not all line equals");
	}
}

void speedProfile(const std::string &name, ge_d** v, const int& column, const int& size, const ge_d& def=1.0, const bool &is_column=true) {
	std::cout<<name;
	for (int i = 1; i < size - 2; i++) {
		if(is_column)
			std::printf("%.3f ", def*v[column][i]);
		else
			std::printf("%.3f ", def*v[i][column]);
	}
	std::cout << std::endl;
}

ge_d conservation(ge_d** v, const int& size) {
	static ge_d last(0.0);
	ge_d tot(0.0);
	for (int j = 0; j < size; j++) {
		for (int i = 0; i < size; i++) {
			tot += v[i][j];
		}
	}
	if (abs(last - tot) >0.0001) {
		Log(LINFO, "Water quantity changed ", last, " ", tot, " diff ", last-tot);
	}
	last = tot;
	return tot;
}

ge_d formula(ge_d const& A, ge_d const& mu, ge_d const& B, ge_d const& dp) {
	return  A; // -mu * B - dp;
}

void CurtilModel::updateVelocities()
{
	static const ge_d dt_factor(2.0), t_max(1.0), print_duv(2.0), big_number(500.0);
	static ge_i frame(0), frame_rate(100);
	static ge_d last(0.0);
	ge_d dt(-1.0), max_v(-1.0), max_u(-1.0), max_p(0.0), tot_p(conservation(_p, _size.h));
	ge_pi max_pu, max_pv;
	//checkEquality(_v, _size.h);
	speedProfile("vc ", _v, 5, _size.h, 100.0);
	//speedProfile("vc ", _v, 5, _size.h, 1.0, false);
	//speedProfile("u ", _u, 1, _size.w, 100000.0, false);
	
	loop() {
		e(_u, i, j) -= e(_nhu, i, j);
		if (abs(e(_u, i, j)) > max_u){
			max_u = abs(e(_u, i, j));
			max_pu = { i, j};
		}

		e(_v, i, j) -= e(_nhv, i, j);
		if (abs(e(_v, i, j)) > max_v){
			max_v = abs(e(_v, i, j));
			max_pv = { i, j };
		}
		if (abs(e(_p, i, j)) > max_p)
			max_p = abs(e(_p, i, j));
	}
	dt = max(max_u, max_v);
	//enforceBoundaryCondition(_u, _v);

	if(frame%frame_rate == 0 || abs(dt-last)>print_duv) 
		Log(LINFO, "Frame ", frame, 
			" u ", max_pu.w, ",", max_pu.h,
			" : ", max_u, " | v ", max_v,
			" : ", max_pv.w, ",", max_pv.h,
			" last ", last,
			" maxp ", max_p,
			" totp ", tot_p);
	if (frame == 29) {
		frame = frame;
	}

		frame++;
	if (dt > 4.0 * last)
		last = 0.0;
	last = dt;

	dt = 1.0 / (dt_factor *dt);
	

	ge_d A, ap1, ap2, B, dp, tmp;
	for (ge_d t = 0.0; t <= t_max && dt > DT_EPSILON; t += dt) {
		safeloop(1) {
			
			if (e(_v, i, j) > big_number || e(_u, i, j) > big_number) {
				tmp = 0.0;
			}
			// Compute up
			// Compute A
			
				tmp = stu(_u, i, j);
				A = tmp * tmp;

				tmp = stu(_u, i-1, j);
				A -= tmp * tmp;

				// A += e(_u, i, j) * e(_v, i, j-1);
				// A -= e(_u, i, j) * e(_v, i, j);
			
			/* Compute A - alternative
			ap1 = e(_v, i + 1, j) + e(_v, i, j);
			ap2 = (ap1 - e(_v, i + 1, j - 1) - e(_v, i, j - 1)) / 2.0;
			ap1 += e(_v, i + 1, j - 1) + e(_v, i, j - 1);
			ap1 /= 4.0;
			A = e(_u, i + 1, j) + e(_u, i - 1, j) - 2.0 * e(_u, i, j);	// du²/dx²
			//A += (e(_u, i, j + 1) - e(_u, i, j - 1)) / 2.0 * ap1;		//  + du/dy*v
			//A += e(_u, i, j) * ap2;										//  + dv/dy*u
			*/
			// Compute B
			B = 4.0 * e(_u, i, j)
				- e(_u, i + 1, j)
				- e(_u, i - 1, j)
				- e(_u, i, j + 1)
				- e(_u, i, j - 1);
			//B = 0.0;
			dp = e(_p, i + 1, j) - e(_p, i, j);
				
			
			// compute final up value

			tmp = e(_u, i, j) +
				dt * (formula(A, _mu, B, dp) - _kappa * e(_u, i, j));
			ifudiff(i + 2) e(_up, i, j) = tmp;
		
			//e(_up, i, j) = 0.0;

			// Compute vp
			// Compute A
			
				tmp = stv(_v, i, j-1);
				A = tmp * tmp;
				tmp = stv(_v, i, j);
				A -= tmp * tmp;

				// A += e(_u, i - 1, j) * e(_v, i, j);
				// A -= e(_u, i, j) * e(_v, i, j);
			
			/* Compute A - alternative
			ap1 = e(_u, i + 1, j) + e(_u, i, j);
			ap2 = (ap1 - e(_u, i + 1, j - 1) - e(_u, i, j - 1)) / 2.0;
			ap1 += e(_u, i + 1, j - 1) + e(_u, i, j - 1);
			ap1 /= 4.0;
			A = e(_v, i, j+1) + e(_v, i, j-1) - 2.0 * e(_v, i, j);	// du²/dx²
			//A += (e(_v, i, j + 1) - e(_v, i, j - 1)) / 2.0 * ap1;		//  + du/dy*v
			//A += e(_v, i, j) * ap2;										//  + dv/dy*u
			//Log(LINFO, e(_v, i, j + 1), " ", e(_v, i, j - 1), " ", e(_v, i, j));
			*/
			// Compute B
			B = 4.0 * e(_v, i, j)
				- e(_v, i + 1, j)
				- e(_v, i - 1, j)
				- e(_v, i, j + 1)
				- e(_v, i, j - 1);
			//B = 0.0;
			dp = e(_p, i, j + 1) - e(_p, i, j);

		
			// compute final up value
			tmp = e(_v, i, j) +
				dt * (formula(A, _mu, B, dp) - _kappa * e(_v, i, j));
			ifvdiff(j + 2) e(_vp, i, j) = tmp;

			 if (e(_vp, i, j) > big_number || e(_up, i, j) > big_number) {
				 tmp = 0.0;
			 }
			 
			//e(_vp, i, j) = 0.0;
		}
		enforceBoundaryCondition(_up, _vp);
		loop() {
			e(_u, i, j) = e(_up, i, j);
			e(_v, i, j) = e(_vp, i, j);
		}
	}


	//checkEquality(_v, _size.h);
}

void CurtilModel::transfertPigment()
{
}

void CurtilModel::transferPigment()
{
}

void CurtilModel::simulateCapilarityFlow()
{
}

void CurtilModel::deleteModel()
{
	Common::Deallocate(_size.w, _size.h, _p);
	Common::Deallocate(_size.w, _size.h, _h);

	Common::Deallocate(_size.w, _size.h, _u);
	Common::Deallocate(_size.w, _size.h, _v);

	Common::Deallocate(_size.w, _size.h, _up);
	Common::Deallocate(_size.w, _size.h, _vp);

	Common::Deallocate(_size.w, _size.h, _nhu);
	Common::Deallocate(_size.w, _size.h, _nhv);

	Common::Deallocate(_size.w, _size.h, _wet);
}

void CurtilModel::generateHeight()
{
	const int iteration(0);
	const ge_d red(4.0), grad_limit(0.5);
	ge_d max_gradu(0.0), max_gradv(0.0);
	bool random(false);

	Log(LINFO, "Generating paper");

	if(random) loop() {
		e(_h, i, j) = (ge_d)(rand() % 100) / 100.0;
	}
	else {
		loop() {
			e(_h, i, j) = (ge_d)(j) / _size.h;
		}
	}

	for(int k=0; k< iteration;k++) {
		loop() {
			if (i > 0) {
				e(_h, i - 1, j) += (e(_h, i, j) - e(_h, i - 1, j))/red;
				e(_h, i, j) += (e(_h, i-1, j) - e(_h, i, j)) / red;
			} if (j > 0) {

				e(_h, i, j-1) += (e(_h, i, j) - e(_h, i, j-1)) / red;
				e(_h, i, j) += (e(_h, i, j-1) - e(_h, i, j)) / red;
			}
		}
	}
	loop() {
		if (e(_h, i, j) < 0.0)
			e(_h, i, j) = 0.0;
		else if (e(_h, i, j) > 1.0)
			e(_h, i, j) = 1.0;
	}
	Log(LINFO, "Computing h grad");
	loop() {
		if (j == _size.h - 2)
			e(_c, i, j) = 2.0*(ge_d)(i) * 1.0 / _size.w - 1.0;
		ifudiff(i+1)
			_nhu[i][j] = gru(_h, i, j);
		ifvdiff(j+1)
			_nhv[i][j] = grv(_h, i, j);
		if (abs(_nhu[i][j]) > max_gradu)
			max_gradu = abs(_nhu[i][j]);
		if (abs(_nhv[i][j]) > max_gradv)
			max_gradv = abs(_nhv[i][j]);
	}
	Log(LINFO, "Max grad ", max_gradu, " ", max_gradv);
}


void CurtilModel::draw()
{
	SDL_Rect rect;
	rect.h = _zoom; rect.w = _zoom;
	loop() {
		/*
		for (int k = 0; k<_zoom; k++)
			for (int l = 0; l<_zoom; l++)
				_scw->setPixelColor(_zoom*i + k, _zoom*j + l, { getFromModel(i, j) });
				*/
		rect.x = _zoom * i; rect.y = _zoom * j;
		 _scw->setRectColor(rect, { getFromModel(i, j) });
		 _scu->setRectColor(rect, { getU(i, j, 3.0) });
		 _scv->setRectColor(rect, { getV(i, j,0.1) });
		 _scp->setRectColor(rect, { getP(i, j, 2.0, 0.5) });
		
	}
	// _scw->updateTexture();
}



Color CurtilModel::getFromModel(const ge_i& i, const ge_i& j) const
{
	ge_d res = 2.0;
	return Color::ColorFromScalar(e(_c, i, j), -res/2.0, res);
}

Color CurtilModel::getU(const ge_i& i, const ge_i& j, const ge_d& res) const
{
	return Color::ColorFromScalar(e(_u, i, j), 0.0, res);
}

Color CurtilModel::getV(const ge_i& i, const ge_i& j, const ge_d& res) const
{
	return Color::ColorFromScalar(e(_v, i, j), 0.00, res);
}

Color CurtilModel::getP(const ge_i& i, const ge_i& j, const ge_d& avg, const ge_d& res) const
{
	return Color::ColorFromScalar(e(_p, i, j), avg -res / 2.0, res);
}


