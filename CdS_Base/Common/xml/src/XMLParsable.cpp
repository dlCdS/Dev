#include "XMLParsable.h"


namespace XML {

	Parsable::Parsable() :
		_fieldVariable(NULL),
		_subCreateFunction(NULL),
		_subVector(NULL),
		_subElement(NULL),
		_subSimple(NULL),
		_postRoutine(NULL),
		_subPolymorphVector(NULL),
		_instanceLoader(NULL),
		_refGetter(NULL),
		_refLoader(NULL),
		_beacon(NULL)
	{
	}


	Parsable::~Parsable()
	{
	}

	void Parsable::XMLLoad(Beacon * beacon, std::list<std::unordered_map<std::string, ParsableReturn>*> &instanceProvider)
	{
		getProvider(instanceProvider);

		associate();

		loadField(beacon);

		loadBeacon(beacon, instanceProvider);

		instanceProvider.pop_back();

		clearAssociation();

		postLoading();
	}

	void Parsable::postLoading()
	{
	}

	void Parsable::XMLParse(Beacon * beacon)
	{
		associate();
		if(_fieldVariable != NULL)
			for (auto f : *_fieldVariable) {
				beacon->addField(f.first, f.second->get());
			}
		if (_beacon != NULL)
			for(auto be : *_beacon) {
				if (_subSimple  != NULL && _subSimple->find(be) != _subSimple->end()) {
					(*_subSimple)[be]->XMLParse(beacon->addBeacon(be));
				}
				if (_subElement  != NULL && _subElement->find(be) != _subElement->end()) {
					(*_subElement)[be]->XMLParse(beacon->addBeacon(be));
				}
				if (_subVector  != NULL && _subVector->find(be) != _subVector->end()) {
					for (auto s : (*_subVector)[be]) {
						s->XMLParse(beacon->addBeacon(be));
					}
				}
				if (_subPolymorphVector  != NULL && _subPolymorphVector->find(be) != _subPolymorphVector->end() ) {
					Beacon *b = beacon->addBeacon(be);
					for (auto s : (*_subPolymorphVector)[be]) {
						s->XMLParse(b->addBeacon(s->XMLName()));
					}
				}
				if (_refGetter != NULL && _refGetter->find(be) != _refGetter->end() ) {
					beacon->addBeacon(be, (*_refGetter)[be]());
				}
		}
		clearAssociation();
	}

	void Parsable::XMLAssociateField(const std::string & field, XML::Field * xmlfield)
	{
		if (_fieldVariable == NULL)
			_fieldVariable = new std::unordered_map<std::string, XML::Field*>();
		_fieldVariable->insert(std::make_pair(field, xmlfield));
	}

	void Parsable::XMLAssociateSubBeacon(const std::string & sub, Parsable * element, ParsableReturn func)
	{
		if (_subCreateFunction == NULL)
			_subCreateFunction = new std::unordered_map<std::string, ParsableReturn >();
		_subCreateFunction->insert(std::make_pair(sub, func));

		if (_subElement == NULL)
			_subElement = new std::unordered_map<std::string, Parsable* >();
		_subElement->insert(std::make_pair(sub, element));

		addBeaconStr(sub);
	}

	void Parsable::XMLAssociateSubBeacon(const std::string & sub, Parsable * element)
	{
		if (_subSimple == NULL)
			_subSimple = new std::unordered_map<std::string, Parsable* >();
		_subSimple->insert(std::make_pair(sub, element));

		addBeaconStr(sub);
	}


	void Parsable::XMLAssociatePolymorph(const std::string & containerName, XML::Parsable * element, ParsableLoader func)
	{

		if (_subPolymorphVector == NULL)
			_subPolymorphVector = new std::unordered_map<std::string, std::vector<Parsable*> >();
		if (_subPolymorphVector->find(containerName) == _subPolymorphVector->end()) {
			_subPolymorphVector->insert(std::make_pair(containerName, std::vector<Parsable*>()));
		}
		
		(*_subPolymorphVector)[containerName].push_back(element);

		if (_instanceLoader == NULL)
			_instanceLoader = new std::unordered_map<std::string, ParsableLoader >();
		_instanceLoader->insert(std::make_pair(containerName, func));

		addBeaconStr(containerName);
	}

	void Parsable::XMLAddInstanceProvider(const std::string & type, ParsableReturn provider)
	{
		if (_instanceProvider == NULL)
			_instanceProvider = new std::unordered_map<std::string, ParsableReturn >();
		if (_instanceProvider->find(type) == _instanceProvider->end())
			_instanceProvider->insert(std::make_pair(type, provider));
		else (*_instanceProvider)[type] = provider;

	}

	void Parsable::XMLAddPostRoutine(const std::string & sub, ParsableReturn routine)
	{
		if (_postRoutine == NULL)
			_postRoutine = new std::unordered_map<std::string, ParsableReturn >();
		_postRoutine->insert(std::make_pair(sub, routine));

		addBeaconStr(sub);
	}

	void Parsable::XMLAddReference(const std::string & sub, ReferenceGetter getter, ReferenceLoader loader)
	{
		if (_refGetter == NULL)
			_refGetter = new std::unordered_map<std::string, ReferenceGetter>();
		_refGetter->insert(std::make_pair(sub, getter));

		if (_refLoader == NULL)
			_refLoader = new std::unordered_map<std::string, ReferenceLoader >();
		_refLoader->insert(std::make_pair(sub, loader));

		addBeaconStr(sub);
	}

	void Parsable::getProvider(std::list<std::unordered_map<std::string, ParsableReturn>*> &instanceProvider)
	{
		for (auto p : *instanceProvider.back()) {
			XMLAddInstanceProvider(p.first, p.second);
		}
		if (_instanceProvider == NULL)
			_instanceProvider = new std::unordered_map<std::string, ParsableReturn >();
		instanceProvider.push_back(_instanceProvider);
	}

	void Parsable::loadField(Beacon * beacon)
	{
		for (auto f : beacon->_field) {
			auto typed = _fieldVariable->find(f._field);
			if (typed != _fieldVariable->end()) {
				typed->second->set(f._value);
			}
			else Log(ERROR, "Could not find associated variable for field '", f._field, "'");
		}
	}

	void Parsable::loadBeacon(Beacon * beacon, std::list< std::unordered_map<std::string, ParsableReturn > *> &instanceProvider)
	{
		for (auto b : beacon->_sub) {
			Log(LOG_LEVEL::LDEBUG, "From ", beacon->_name, " building ", b._name);
			if (subcreateFunction(b, instanceProvider)) { ; }
			else if (subElement(b, instanceProvider)) { ; }
			else if (subSimple(b, instanceProvider)) { ; }
			else if (subPolymorph(b, instanceProvider)) { ; }
			else if (subReference(b)) { ; }
			else Log(ERROR, "Could not find association for beacon '", b._name, "'");
		}
	}

	bool Parsable::subcreateFunction(Beacon &b, std::list<std::unordered_map<std::string, ParsableReturn>*> &instanceProvider)
	{
		if (_subCreateFunction != NULL && _subCreateFunction->find(b._name) != _subCreateFunction->end()) {
			Parsable *ret = ((*_subCreateFunction)[b._name])(NULL);
			if (ret != NULL)
				ret->XMLLoad(&b, instanceProvider);
			postRoutine(b._name);
			return true;
		}
		return false;
	}

	bool Parsable::subElement(Beacon &b, std::list<std::unordered_map<std::string, ParsableReturn>*> &instanceProvider)
	{
		if (_subElement  != NULL && _subElement->find(b._name) != _subElement->end()) {
			(*_subElement)[b._name]->XMLLoad(&b, instanceProvider);
			postRoutine(b._name);
			return true;
		}
		return false;
	}

	bool Parsable::subSimple(Beacon &b, std::list<std::unordered_map<std::string, ParsableReturn>*> &instanceProvider)
	{
		if (_subSimple  != NULL && _subSimple->find(b._name) != _subSimple->end()) {
			(*_subSimple)[b._name]->XMLLoad(&b, instanceProvider);
			postRoutine(b._name);
			return true;
		}
		return false;
	}

	bool Parsable::subPolymorph(Beacon  &b, std::list<std::unordered_map<std::string, ParsableReturn>*> &instanceProvider)
	{
		if (_subPolymorphVector  != NULL && _subPolymorphVector->find(b._name) != _subPolymorphVector->end()) {
			for (auto s : b._sub) {
				if (_instanceProvider->find(s._name) != _instanceProvider->end()) {
					Parsable *ret = ((*_instanceProvider)[s._name])(NULL);
					(*_instanceLoader)[b._name](ret);
					if (ret != NULL)
						ret->XMLLoad(&s, instanceProvider);
					postRoutine(b._name);
				}
				else Log(ERROR, "Could not find class provider for beacon '", s._name, "'");
			}
			return true;
		}
		return false;
	}

	bool Parsable::subReference(Beacon & b)
	{
		if (_refLoader  != NULL && _refLoader->find(b._name) != _refLoader->end()) {
			(*_refLoader)[b._name](&b._value);
			postRoutine(b._name);
			return true;
		}
		return false;
	}

	void Parsable::postRoutine(const std::string & sub)
	{
		if (_postRoutine  != NULL && _postRoutine->find(sub) != _postRoutine->end()) {
			(*_postRoutine)[sub](NULL);
		}
	}

	void Parsable::addBeaconStr(const std::string & str)
	{
		if (_beacon == NULL)
			_beacon = new std::vector<std::string>();
		_beacon->push_back(str);
	}

	void Parsable::clearAssociation()
	{

		if (_fieldVariable != NULL){
			for (auto t : *_fieldVariable)
				delete t.second;
			_fieldVariable->clear();
			delete _fieldVariable;
			_fieldVariable = NULL;
		}

		if(_subCreateFunction != NULL){
			_subCreateFunction->clear();
			delete _subCreateFunction;
			_subCreateFunction = NULL;
		}

		if (_subVector  != NULL){
			_subVector->clear();
			delete _subVector;
			_subVector = NULL;
		}

		if (_subElement  != NULL){
			_subElement->clear();
			delete _subElement;
			_subElement = NULL;
		}

		if (_subSimple  != NULL){
			_subSimple->clear();
			delete _subSimple;
			_subSimple = NULL;
		}

		if (_postRoutine  != NULL){
			_postRoutine->clear();
			delete _postRoutine;
			_postRoutine = NULL;
		}

		if (_subPolymorphVector  != NULL){
			_subPolymorphVector->clear();
			delete _subPolymorphVector;
			_subPolymorphVector = NULL;
		}

		if (_instanceLoader  != NULL){
			_instanceLoader->clear();
			delete _instanceLoader;
			_instanceLoader = NULL;
		}

		if (_refGetter  != NULL){
			_refGetter->clear();
			delete _refGetter;
			_refGetter = NULL;
		}

		if (_refLoader  != NULL){
			_refLoader->clear();
			delete _refLoader;
			_refLoader = NULL;
		}

		if (_beacon != NULL){
			_beacon->clear();
			delete _beacon;
			_beacon = NULL;
		}
	}

}