#pragma once
#include <time.h>
#include <string>
#include <unordered_map>
#include <map>
#include <iostream>
#include <list>
#include <unordered_set>
#include <thread>
#include <sstream>
#include "Logger.h"

#define THREAD_ID  std::thread::get_id()

typedef unsigned long long clock_f;
struct ClockCount {
	clock_f last=clock(), tot = 0, count=0;
};

struct Stack {
	std::list<std::string> stack;
	std::string id;
	Stack() : id("") {}

	void push(const std::string& function, const std::thread::id& t_id) {
		if (id == "") {
			std::stringstream ss;
			ss << t_id;
			id = "thread_";
			id += ss.str();
			stack.push_back(id);
			
		}
		stack.push_back(function);
		id += "/" + function;
	}
	void pop() {
		id = id.substr(0, id.size() - stack.back().size() - 1);
		stack.pop_back();
	}
	int check(const std::string function) {
		if (function != stack.back()) {
			return -1;
		}
		else return 0;
	}
};

class Clocks
{
public:
	Clocks();
	~Clocks();

	static void addClock(const std::string &s, const std::thread::id& id= _default);
	static void prepare(const std::string& s, const std::thread::id& id = _default);
	static void start(const std::string &s, const std::thread::id& id= _default);
	static clock_t stop(const std::string &s, const std::thread::id& id= _default);
	static void report();

private:
	static std::string getStackId(const std::string& function, const std::thread::id& id);
	static std::string popStackId(const std::string& function, const std::thread::id& id);
	static std::string prepareStackId(const std::string& function, const std::thread::id& id);

	static std::map<std::string, ClockCount> _clocks;
	static std::unordered_set<std::string> _logged;
	static std::unordered_map<std::thread::id, Stack> _thread_stack;
	static std::thread::id _default;

	static clock_f _execTime;
};

