#pragma once
#include <VirtualModel.h>

// stagered grid definition
#define usize _size.w-1
#define vsize _size.h-1
#define stu(f, i, j) ((f[i][j] + f[i+1][j])/2.0)
#define stv(f, i, j) ((f[i][j] + f[i][j+1])/2.0)

// grad diff
#define gru(f, i, j) ((f[i+1][j] - f[i][j])/2)
#define grv(f, i, j) ((f[i][j+1] - f[i][j])/2)

// grid size checkup
#define ifudiff(i) if(i < _size.w)
#define ifvdiff(j) if(j < _size.h)
#define ifinf(i) if(i > 0)


#define safeloop(n) for(ge_i i=1; i<_size.w-n; i++) for(ge_i j=1; j<_size.h-n; j++)
#define uloop() for(ge_i i=0; i<_size.w-1; i++) for(ge_i j=0; j<_size.h; j++)
#define vloop() for(ge_i i=0; i<_size.w; i++) for(ge_i j=0; j<_size.h-1; j++)

// Other defines
#define DT_EPSILON 0.01

class CurtilModel : public VirtualModel
{
public:
	CurtilModel();
	~CurtilModel();

	void setUVPWidget(SetColourWidget* scu, SetColourWidget* scv, SetColourWidget* scp);

	void generate(const ge_d &mu, const ge_d &kappa, 
		const ge_d &ksi, const ge_d &tau);

protected:
	
	void initialize();

	virtual void mainLoop();

	void moveWater();
	void updateVelocities();
	void enforceBoundaryCondition(ge_d **u, ge_d **v);
	void relaxDivergence();

	void transfertPigment();
	void transferPigment();
	void simulateCapilarityFlow();

	void deleteModel();

	void generateHeight();

	virtual void draw();


	SetColourWidget * _scu, * _scv, * _scp;
	Color getFromModel(const ge_i &i, const ge_i &j) const;

	Color getU(const ge_i& i, const ge_i& j, const ge_d&res) const;
	Color getV(const ge_i& i, const ge_i& j, const ge_d& res) const;
	Color getP(const ge_i& i, const ge_i& j, const ge_d&avg, const ge_d& res) const;

	std::list<ge_pi> _notwet;
	ge_pi ***_wet;

	// Model Data
	ge_d **_p,	// water
		**_u,	// u vel
		**_up,	// u temp
		**_v,	// v vel
		**_vp,	// v temp
		**_h,	// height
		**_nhu,	// nabla hu
		**_nhv,	// nabla hv
		**_c,	// color
		_mu,
		_kappa,
		_ksi,
		_tau;
};

