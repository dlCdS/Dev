#pragma once
#include <unordered_set>
#include <unordered_map>
#include <list>
#include <typeinfo>
#include "Common.h"

#define DEFAULT_XML_HEADER "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
#define DEFAULT_TAB "  "




namespace XML {
	struct Couple {
		std::string _field;
		std::string  _value;
		Couple(const std::string &f, const std::string &v) : _field(f), _value(v) {}
	};

	struct Beacon {
		Beacon(const std::string &name, Beacon *parent) : _name(name), _prev(parent) {}
		Beacon(const std::string &name, Beacon *parent, const std::string &value) : _name(name), _prev(parent), _value(value) {}
		Beacon *_prev;
		std::string _name;
		std::string _value;
		std::vector<Beacon> _sub;
		std::vector<Couple> _field;
		std::unordered_set<std::string> _knownFields;

		void addField(const std::string field, const std::string value) {
			Log(LOG_LEVEL::LDEBUG, "Added Couple '", field, "' : '", value, "' to '", _name, "'");
			_field.push_back(Couple(field, value));
		}

		Beacon *addBeacon(const std::string &name) {
			_sub.push_back(Beacon(name, this));
			Log(LOG_LEVEL::LDEBUG, "Added Beacon '", name, "' to '", _name, "'");
			return &_sub.back();
		}

		Beacon *addBeacon(const std::string &name, const std::string &value) {
			_sub.push_back(Beacon(name, this, value));
			Log(LOG_LEVEL::LDEBUG, "Added Beacon '", name, "' to '", _name, "'");
			return &_sub.back();
		}

		void dump() {
			_sub.clear();
			_field.clear();
			_value = "";
		}

		void indent(const int &offset, std::fstream &file) {
			for (int i = 0; i < offset; i++)
				file << DEFAULT_TAB;
		}

		void save(const int &offset, std::fstream &file) {
			indent(offset, file); file << '<' << _name;
			for (auto f : _field)
				file << ' ' << f._field << "=\"" << f._value << '"';
			if (_sub.size() > 0) {
				file << '>' << std::endl;
				for (auto b : _sub)
						b.save(offset + 1, file);
				indent(offset, file); file << "</" << _name << ">" << std::endl;
			}
			else if (_value == "") {
				file << "/>" << std::endl;
			}
			else {
				file << '>' << _value << "</" << _name << ">" << std::endl;
			}
		}
	};

	struct Info {
		std::string _name;
		std::unordered_map<std::string, Couple> _field;
	};

	class Parser
	{
		enum MODE { NONE, OPEN, BEACON, S_CLOSE, P_ERROR, E_CLOSE, READ, READ_STR, COUPLE, COMMENT, INFO };
		const char *MODE_STR[11] = { "NONE", "OPEN", "BEACON", "S_CLOSE", "ERROR", "E_CLOSE", "READ", "READ_STR", "COUPLE", "COMMENT", "INFO" };
	public:
		Parser();
		virtual ~Parser();

		static Parser *readFile(const std::string &filename);
		static bool skip(const char &c);

		void save(const std::string &filename);

		template<typename T>
		static T cast(const std::string &str);

		template<typename T>
		static std::string cast(const T &value);

		Beacon *getBase();

	protected:

		void useChar(const char &c);
		void none(const char &c);
		void open(const char &c);
		void beacon(const char &c);
		void error(const char &c);
		void s_close(const char &c);
		void e_close(const char &c);
		void read(const char &c);
		void read_str(const char &c);
		void couple(const char &c);
		void comment(const char &c);
		void info(const char &c);

		void log(const char & c) const;

		void createBeacon();
		bool closeBeacon();
		void addBeaconField();
		bool addCouple();

		bool _error;
		MODE _mode;
		Beacon *_base, *_cur;
		std::string _str, _field, _header, _tab;
		std::list<std::string> _stack;
	};

	template<typename T>
	inline T Parser::cast(const std::string & str)
	{
		return Common::Cast<T>(str);
	}

	template<typename T>
	inline std::string Parser::cast(const T &value)
	{
		return Common::Cast<T>(value);
	}

	class Field {
	public:
		Field() {}
		virtual void set(const std::string &value) = 0;
		virtual std::string get() const = 0;
		virtual void *getVar() = 0;
		virtual Field *getCopy() = 0;
	};

	class Integer : public Field {
	public:
		Integer(int *val) : _val(val) {}
		virtual void set(const std::string &value) { Common::SafeCast<int>(*_val, value); }
		virtual std::string get() const { return Parser::cast<int>(*_val); }
		virtual void *getVar() { return _val; }
		virtual Field *getCopy() { return new Integer(_val); }
	private:
		int *_val;
	};

	class UInteger8 : public Field {
	public:
		UInteger8(uint8_t *val) : _val(val) {}
		virtual void set(const std::string &value) { 
			unsigned int ui;
			Common::SafeCast<unsigned int>(ui, value);
			*_val = ui & 0x000000ff;
		}
		virtual std::string get() const { return Parser::cast<unsigned int>(*_val); }
		virtual void *getVar() { return _val; }
		virtual Field *getCopy() { return new UInteger8(_val); }
	private:
		uint8_t *_val;
	};

	class UInteger : public Field {
	public:
		UInteger(unsigned int *val) : _val(val) {}
		virtual void set(const std::string &value) { Common::SafeCast<unsigned int>(*_val, value); }
		virtual std::string get() const { return Parser::cast<unsigned int>(*_val); }
		virtual void *getVar() { return _val; }
		virtual Field *getCopy() { return new UInteger(_val); }
	private:
		unsigned int *_val;
	};

	class Double : public Field {
	public:
		Double(ge_d*val) : _val(val) {}
		virtual void set(const std::string &value) { Common::SafeCast<ge_d>(*_val, value); }
		virtual std::string get() const { return Parser::cast<ge_d>(*_val); }
		virtual void *getVar() { return _val; }
		virtual Field *getCopy() { return new Double(_val); }
	private:
		ge_d *_val;
	};

	class Bool : public Field {
	public:
		Bool(bool *val) : _val(val) {}
		virtual void set(const std::string &value) { Common::SafeCast<bool>(*_val, value); }
		virtual std::string get() const { return Parser::cast<bool>(*_val); }
		virtual void *getVar() { return _val; }
		virtual Field *getCopy() { return new Bool(_val); }
	private:
		bool *_val;
	};

	class String : public Field {
	public:
		String(std::string *val) : _val(val) {}
		virtual void set(const std::string &value) { *_val = value; }
		virtual std::string get() const { return *_val; }
		virtual void *getVar() { return _val; }
		virtual Field *getCopy() { return new String(_val); }
	private:
		std::string *_val;
	};
}