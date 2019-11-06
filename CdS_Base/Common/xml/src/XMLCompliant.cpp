#include "XMLCompliant.h"


namespace XML {

	Compliant::Compliant(const std::string &baseName) :
		_base(this, baseName)
	{
	}


	Compliant::~Compliant()
	{
	}

	void Compliant::load(Parser * parser)
	{
		_base.load(parser->getBase());
	}

	void Compliant::load(const std::string & filename)
	{
		Parser *parser = Parser::readFile(filename);
		if(parser != NULL)
			load(parser);
		delete parser;
	}

	void Compliant::save(const std::string & filename)
	{
		Parser *parser = new Parser();
		_base.parse(parser->getBase());
		parser->save(filename);
		delete parser;
	}

	Base::Base(Parsable *parent, const std::string &baseName) : _parent(parent), _baseName(baseName) {}

	void Base::parse(Beacon *b)
	{
		XMLParse(b);
	}

	void Base::load(Beacon * b)
	{
		std::list<std::unordered_map<std::string, ParsableReturn > * > provider;
		if (_instanceProvider == NULL)
			_instanceProvider = new std::unordered_map<std::string, ParsableReturn >();
		provider.push_back(_instanceProvider);
		XMLLoad(b, provider);
		provider.pop_back();
		if (provider.size() > 0) Log(LOG_LEVEL::LWARNING, "Class Provider stack ended with unexpected functions");
	}

	const std::string & Base::getName() const
	{
		return _baseName;
	}

	void Base::associate()
	{
		XMLAssociateSubBeacon(_baseName, _parent, SubBeaconLoadFunction(Base, add));
	}

	Parsable * Base::add(void * v)
	{
		return _parent;
	}
}