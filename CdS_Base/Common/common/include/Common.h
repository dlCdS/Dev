#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <time.h>
#include "Logger.h"

#define __COMPILE_WINDOWS__
#ifdef __COMPILE_WINDOWS__
#include <Windows.h>
#include <Commdlg.h>
//#include <shobjidl.h>
#endif

namespace Common {
	template<typename T>
	static T Cast(const std::string &str) {
		std::stringstream stream;
		T value;
		stream << str;
		stream >> value;
		if (stream.fail()) {
			Log(LOG_LEVEL::LERROR, "XMLCast failed to cast ", str, " to ", typeid(value).name());
		}
		return value;
	}

	template<typename T>
	static bool SafeCast(T &dest, const std::string &str) {
		std::stringstream stream;
		T value;
		stream << str;
		stream >> value;
		if (stream.fail()) {
			Log(LOG_LEVEL::LERROR, "XMLCast failed to cast ", str, " to ", typeid(value).name());
			return false;
		}
		dest = value;
		return true;
	}

	template<typename T>
	static std::string Cast(const T &value) {
		std::stringstream stream;
		std::string str;
		stream << value;
		if (stream.fail()) {
			Log(LOG_LEVEL::LERROR, "XMLCast failed to interpret ", value, " as sstream");
		}
		stream >> str;
		if (stream.fail()) {
			Log(LOG_LEVEL::LERROR, "XMLCast failed to cast ", value);
		}
		return str;
	}

	template<typename T>
	static bool SafeCast(std::string &dest, const T &value) {
		std::stringstream stream;
		std::string str;
		stream << value;
		if (stream.fail()) {
			Log(LOG_LEVEL::LERROR, "XMLCast failed to interpret ", value, " as sstream");
			return false;
		}
		stream >> str;
		if (stream.fail()) {
			Log(LOG_LEVEL::LERROR, "XMLCast failed to cast ", value);
			return false;
		}
		dest = str;
		return true;
	}


	template<typename T>
	static T** Allocate(const int &size_x, const int &size_y) {
		T **ptr = NULL;
		ptr = new T*[size_x];
		for (int i = 0; i < size_x; i++)
			ptr[i] = new T[size_y];
		return ptr;
	}

	template<typename T>
	static void Deallocate(const int &size_x, const int &size_y, T **ptr) {
		for (int i = 0; i < size_x; i++)
			delete[] ptr[i];
		delete[] ptr;
		ptr = NULL;
	}
}

class Workspace {
public:
	static std::string Path(const std::string &path = "");
	static std::string GetCurrent();
	static std::string StartPath();
	static std::string Explore();
	static std::string GetFile();
	static void SetPath(const std::string &path);
	static void Directory(const std::string &path);
	static std::string StripFile(const std::string &str);

private:
	Workspace();
	std::string _path, _startPath;
	static Workspace _singleton;
};

#define COMMON_M_PI 3,1415926535

#define ge_d float
#define ge_i int
#define ge_c char

template<typename T>
struct ge_p {
	T h, w;
	ge_p() : h(0), w(0) {}
	ge_p(const T &_w, const T &_h) : h(_h), w(_w) {}
};

#define EPSILON 0.01
template<typename T>
struct ge_v {
	T x, y;
	ge_v() : x(0), y(0) {}
	ge_v(const T &_x, const T &_y) : x(_x), y(_y) {}
	void operator=(const ge_v &v) {
		x = v.x; y = v.y;
	}
	const ge_v operator+(const ge_v &v) {
		return { x + v.x, y + v.y };
	}

	void operator+=(const ge_v &v) {
		x += v.x, y += v.y ;
	}

	const T operator*(const ge_v &v) {
		return  x * v.y - y * v.x;
	}

	const ge_v operator*(const T &s) {
		return  { s*x, s*y };
	}

	const ge_v operator-(const ge_v &v) {
		return { x - v.x, y - v.y };
	}

	void operator-=(const ge_v &v) {
		x -= v.x; y -= v.y;

	}

	const T norm() {
		return sqrt(x*x + y * y);
	}

	void normalize() {
		T n = norm();
		if (n > EPSILON) {
			x /= n;
			y /= n;
		}
		else {
			randomVector();
		}
	}

	void randomVector() {
		T theta = (T)(rand() % 1000);
		theta *= 2.0*COMMON_M_PI / 1000;
		x = cos(theta);
		y = sin(theta);
	}
};


#define ge_vi ge_v<ge_i>
#define ge_vd ge_v<ge_d>
#define ge_pi ge_p<ge_i>
#define ge_pd ge_p<ge_d>

template<typename T>
struct square {
	ge_p<T> dim, pos;
	square() : dim(0, 0), pos(0, 0) {}
	square(const ge_p<T> &_p, const ge_p<T> &_d) : dim(_d), pos(_p) {}
	square(const T &_pw, const T &_ph, const T &_dw, const T &_dh) :
		dim(ge_p<T>(_dw, _dh)), pos(ge_p<T>(_pw, _ph)) {}

};


#define square_i square<ge_i>
#define square_d square<ge_d>

namespace Math {
	namespace {
		std::vector<ge_i> Prime;
	}
	inline bool Devidable(const ge_i &v1, const ge_i &v2) {
		ge_d div((ge_d)v1 / v2);
		if (div == (ge_d)(v1 / v2))
			return true;
		return false;
	}
	inline void ComputePrimes(const ge_i &to) {
		if (Prime.size() == 0)
			Prime.push_back(2);
		
		for (ge_i i = Prime[Prime.size() - 1] + 1; i <= to; i++) {
			bool isPrime(true);
			for (auto p = Prime.begin(); p != Prime.end() && isPrime; ++p) {
				if (Devidable(i, *p))
					isPrime = false;
			}
			if (isPrime)
				Prime.push_back(i);
		}
	}
	inline ge_i PGCD(const ge_i &v1, const ge_i &v2) {
		ge_i pgcd(1), tv1(v1), tv2(v2);
		ComputePrimes(tv1);
		ComputePrimes(tv2);
		for (auto p = Prime.begin(); p != Prime.end() && *p <= tv1 && *p < tv2;) {
			if (Devidable(tv1, *p) && Devidable(tv2, *p)) {
				tv1 /= *p;
				tv2 /= *p;
				pgcd *= *p;
			}
			else ++p;
		}
		return pgcd;
	}
	template<typename T>
	inline T ScalarProd(const T *v1, const T* v2, const int &size) {
		T res = 0;
		for (int i = 0; i < size; i++)
			res += v1[i]*v2[i];
		return res;
	}
}