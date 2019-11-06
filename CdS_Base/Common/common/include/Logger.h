#pragma once
#include <iostream>
#include <string>

enum LOG_LEVEL {LDEBUG=0, LINFO, LWARNING, LERROR, LFATAL};

namespace {

	LOG_LEVEL log_level = LINFO;

	const char *LOG_LEVEL_STR[5] = { "DEBUG", "INFO", "WARNING", "ERROR", "FATAL" };

	void Log() {
		std::cout << std::endl;
	}

	template<typename First, typename ... Strings>
	void Log(const First &arg, const Strings &... rest) {
		std::cout << arg;
		Log(rest...);
	}
}

inline void SetLogLevel(const LOG_LEVEL &level) { log_level = level; }

template<typename ... Strings>
void Log(const LOG_LEVEL &level, const Strings& ... rest) {
	if (level >= log_level) {
		std::cout << "LOG[" << LOG_LEVEL_STR[level] << "] \t";
		Log(rest...);
	}
}