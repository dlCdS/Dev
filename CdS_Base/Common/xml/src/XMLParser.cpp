#include "XMLParser.h"

#define BASE_NAME "base_beacon_name"

namespace XML{
	Parser::Parser() :
		_error(false),
		_mode(NONE),
		_base(NULL),
		_cur(NULL),
		_str(""),
		_header(DEFAULT_XML_HEADER),
		_tab(DEFAULT_TAB)
	{
		_base = new Beacon(BASE_NAME, NULL);
		_cur = _base;
	}


	Parser::~Parser()
	{
		_base->dump();
		delete _base;
	}

	Parser * Parser::readFile(const std::string & filename)
	{
		Parser *parser = NULL;
		std::fstream file(filename.c_str(), std::ios::in);
		if (file) {
			parser = new Parser();
			bool exit(false);
			char c;
			while (!exit && !file.eof() && !parser->_error) {
				file.get(c);
				if (!skip(c))
					parser->useChar(c);
			}
		} file.close();
		return parser;
	}

	bool Parser::skip(const char & c)
	{
		if (c == '\n')
			return true;
		if (c == '\r')
			return true;
		return false;
	}

	void Parser::save(const std::string & filename)
	{
		if (_base != NULL) {
			std::fstream file(filename.c_str(), std::ios::out | std::ios::trunc);
			if (file) {
				file << _header << std::endl;
				for (auto b : _base->_sub)
						b.save(0, file);
				file.close();
			}
			else  Log(LOG_LEVEL::LERROR, "Could not open file ", filename);
		}
		else Log(LOG_LEVEL::LERROR, "Parser has no data");
	}

	Beacon * Parser::getBase()
	{
		return _base;
	}

	void Parser::useChar(const char & c)
	{
		switch (_mode)
		{
		case Parser::NONE:
			none(c);
			break;
		case Parser::OPEN:
			open(c);
			break;
		case Parser::BEACON:
			beacon(c);
			break;
		case Parser::S_CLOSE:
			s_close(c);
			break;
		case Parser::P_ERROR:
			error(c);
			break;
		case Parser::E_CLOSE:
			e_close(c);
			break;
		case Parser::READ:
			read(c);
			break;
		case Parser::READ_STR:
			read_str(c);
			break;
		case Parser::COUPLE:
			couple(c);
			break;
		case Parser::COMMENT:
			comment(c);
			break;
		case Parser::INFO:
			info(c);
			break;
		default:
			log(c);
			_mode = P_ERROR;
			break;
		}
	}

	void Parser::none(const char & c)
	{
		switch (c)
		{
		case ' ':
			break;
		case '\t':
			break;
		case '<':
			_mode = OPEN;
			break;
		default:
			log(c);
			_mode = P_ERROR;
			break;
		}
	}

	void Parser::open(const char & c)
	{
		switch (c)
		{
		case '?':
			_mode = INFO;
			break;
		case '!':
			_mode = COMMENT;
			break;
		case '/':
			_mode = S_CLOSE;
			break;
		case ' ':
			log(c);
			_mode = P_ERROR;
			break;
		default:
			_str += c;
			_mode = BEACON;
			break;
		}
	}

	void Parser::beacon(const char & c)
	{
		switch (c)
		{
		case '>':
			createBeacon();
			_mode = READ;
			break;
		case '/':
			_mode = E_CLOSE;
			break;
		case ' ':
			createBeacon();
			_mode = COUPLE;
			break;
		default:
			_str += c;
			break;
		}
	}

	void Parser::error(const char & c)
	{
		_error = true;
		_base->dump();
	}

	void Parser::s_close(const char & c)
	{
		switch (c)
		{
		case '<':
			log(c);
			_mode = P_ERROR;
			break;
		case '/':
			log(c);
			_mode = P_ERROR;
			break;
		case '>':
			if (closeBeacon())
				_mode = NONE;
			else {
				log(c);
				_mode = P_ERROR;
			}
			break;
		case ' ':
			log(c);
			_mode = P_ERROR;
			break;
		default:
			_str += c;
			break;
		}
	}


	void Parser::e_close(const char & c)
	{
		switch (c)
		{
		case '<':
			log(c);
			_mode = P_ERROR;
			break;
		case '/':
			log(c);
			_mode = P_ERROR;
			break;
		case '>':
			if (closeBeacon())
				_mode = NONE;
			else {
				log(c);
				_mode = P_ERROR;
			}
			break;
		case ' ':
			log(c);
			_mode = P_ERROR;
			break;
		default:
			log(c);
			_mode = P_ERROR;
			break;
		}
	}

	void Parser::read(const char & c)
	{
		switch (c)
		{
		case ' ':
			break;
		case '\t':
			break;
		case '<':
			addBeaconField();
			_mode = OPEN;
			break;
		case '/':
			log(c);
			_mode = P_ERROR;
			break;
		case '>':
			log(c);
			_mode = P_ERROR;
			break;
		default:
			_str += c;
			break;
		}
	}

	void Parser::read_str(const char & c)
	{
		switch (c)
		{
		case '"':
			_mode = COUPLE;
			break;
		default:
			_str += c;
			break;
		}
	}

	void Parser::couple(const char & c)
	{
		switch (c)
		{
		case '<':
			log(c);
			_mode = P_ERROR;
			break;
		case '/':
			if (addCouple())
				_mode = E_CLOSE;
			else {
				log(c);
				_mode = P_ERROR;
			}
			break;
		case '>':
			if (addCouple())
				_mode = NONE;
			else {
				log(c);
				_mode = P_ERROR;
			}
			break;
		case ' ':
			if (addCouple())
				_mode = COUPLE;
			else {
				log(c);
				_mode = P_ERROR;
			}
			break;
		case '"':
			_mode = READ_STR;
			break;
		case '=':
			_field = _str;
			_str = "";
			break;
		default:
			_str += c;
			break;
		}
	}

	void Parser::comment(const char & c)
	{
		switch (c)
		{
		case '>':
			_mode = NONE;
			break;
		default:
			break;
		}
	}

	void Parser::info(const char & c)
	{
		switch (c)
		{
		case '>':
			_mode = NONE;
			break;
		default:
			break;
		}
	}

	void Parser::log(const char & c) const
	{
		Log(LOG_LEVEL::LERROR, "XML Parser with beacon '", _cur->_name, "', str '", _str, "' mode '", MODE_STR[_mode], "' with char '", c, "'");
	}

	void Parser::createBeacon()
	{
		_cur = _cur->addBeacon(_str);
		_stack.push_back(_str);
		_str = "";
	}

	bool Parser::closeBeacon()
	{
		if (_cur->_name == _stack.back()) {
			Log(LOG_LEVEL::LDEBUG, "Close Beacon '", _cur->_name, "'");
			_cur = _cur->_prev;
			_stack.pop_back();
			_str = "";
			return true;
		}
		else {
			Log(LOG_LEVEL::LDEBUG, "Could not close Beacon '", _cur->_name, "' stack '", _stack.back(), "'");
			_str = "";
			return false;
		}
	}

	void Parser::addBeaconField()
	{
		Log(LOG_LEVEL::LINFO, "Added field '", _str, "' to '", _cur->_name, "'");
		_cur->_value = _str;
		_str = "";
	}

	bool Parser::addCouple()
	{
		if (_field == "") {
			if (_str == "")
				return true;
			else return false;
		}
		else {
			_cur->addField(_field, _str);
			_str = ""; _field = "";
			return true;
		}
	}
}