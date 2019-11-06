#include "Clocks.h"

std::map<std::string, ClockCount> Clocks::_clocks= std::map<std::string, ClockCount>();
std::unordered_set<std::string> Clocks::_logged = std::unordered_set<std::string>();
std::unordered_map<std::thread::id, Stack > Clocks::_thread_stack = std::unordered_map<std::thread::id, Stack >();
clock_f Clocks::_execTime=0;
std::thread::id Clocks::_default = std::thread::id();

Clocks::Clocks()
{
}


Clocks::~Clocks()
{
}

void Clocks::addClock(const std::string & s, const std::thread::id& id)
{
	if(_clocks.find(s) == _clocks.end())
		_clocks.insert(std::make_pair(s, ClockCount()));
}

void Clocks::prepare(const std::string& s, const std::thread::id& id)
{
	Stack* stack = &_thread_stack[id];
	std::string stack_id = stack->id;
	if (stack->check(s) != 0) {
		stack->push(s, id);
	}
}

void Clocks::start(const std::string & s, const std::thread::id &id)
{
	clock_f t = clock_f(clock());
	std::string stack_id = getStackId(s, id);
	_clocks[stack_id].last = clock_f(clock());
	_execTime += clock() - t;
}

clock_t Clocks::stop(const std::string & s, const std::thread::id& id)
{
	clock_t t = clock();
	std::string stack_id = popStackId(s, id);
	_clocks[stack_id].tot += clock_f(clock()) - _clocks[stack_id].last;
	_clocks[stack_id].count++;
	_execTime += clock() - t;
	return clock() - _clocks[stack_id].last;
}

std::vector<std::string> split(const std::string& s, char delimiter)
{
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delimiter))
	{
		tokens.push_back(token);
	}
	return tokens;
}

std::string indent(const int& n) {
	std::string ind("");
	for (int i = 0; i < n; i++)
		ind += "|  ";
	ind += "|";
	return ind;
}

std::string titleIndent(const int& n) {
	std::string ind("");
	for (int i = 0; i < n; i++)
		ind += "|  ";
	ind += "+- ";
	return ind;
}

int compare(const std::vector<std::string>& ref, std::vector<std::string>& stack) {
	int c(0);
	for (int i = 0; i < ref.size() && i<stack.size(); i++) {
		if (ref[i] != stack[i]) {
			i = ref.size();
		}
		c = i;
	}
	return c;
}

std::string relevantName(const std::vector<std::string>& ref, std::vector<std::string>& stack) {
	std::string name("");
	bool equal(true);
	for (int i = 0; i < stack.size(); i++) {
		if (i >= ref.size()) {
			equal = false;
		}
		else if (ref[i] != stack[i]) {
			equal = false;
		}

		if(!equal) 
			name += "/" + stack[i];
		
	}
	return name;
}

void Clocks::report()
{
	std::vector<std::string> stack, ref;
	int ind;
	std::string useful_name, indentation;
	std::cout << "Clock report: self execution time: " << 1000000 * _execTime / CLOCKS_PER_SEC << std::endl << std::endl;
	for (auto a = _clocks.begin(); a != _clocks.end(); ++a) {
		stack = split(a->first, '/');
		ind = compare(ref, stack);
		useful_name = relevantName(ref, stack);
		indentation= indent(ind);
		if (a->second.count > 0) {
			std::cout << titleIndent(ind);
			std::cout << a->first << ":" << std::endl;
			std::cout << indentation << "total: " << 1000000 * (a->second.tot) / CLOCKS_PER_SEC << " micro s" << std::endl;
			std::cout << indentation << "average: " << 1000000 * (a->second.tot) / a->second.count / CLOCKS_PER_SEC << " micro s" << std::endl;
			std::cout << indentation << "used: " << a->second.count << std::endl << std::endl;
		}
		else {
			std::cout << titleIndent(ind);
			std::cout << a->first << ": unused" << std::endl << std::endl;
		}

		ref = stack;
	}
}

std::string Clocks::getStackId(const std::string& function, const std::thread::id& id)
{
	Stack* stack = &_thread_stack[id];
	stack->push(function, id);
	return stack->id;
}

std::string Clocks::popStackId(const std::string& function, const std::thread::id& id)
{
	Stack* stack = &_thread_stack[id];
	std::string stack_id = stack->id;
	if (stack->check(function) != 0) {
		if (_logged.find(stack_id) == _logged.end()) {
			Log(LWARNING, "Clocks failed to close function ", function, " in stack ", stack_id);
			_logged.insert(stack_id);
		}
	}
	_thread_stack[id].pop();
	return stack_id;
}

std::string Clocks::prepareStackId(const std::string& function, const std::thread::id& id)
{
	return std::string();
}
