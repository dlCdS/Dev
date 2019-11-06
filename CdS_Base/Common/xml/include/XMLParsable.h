#pragma once
#include "XMLParser.h"
#include <functional>


#define SubBeaconLoadFunction(class,function) std::bind(&class::function, this, std::placeholders::_1)
#define SubBeaconGetReference(class,function,instance) std::bind(&class::function, std::cref(instance))
#define VoidedCallbackFunction(class,function) std::bind(&class::function, this, std::placeholders::_1)

namespace XML {

	class Parsable;
	typedef void (Parsable::*XMLParsableFn)(Beacon *);
	typedef void(*VoidBeaconFn)(Beacon *);
	typedef std::function<Parsable*(void *v)> ParsableReturn;
	typedef std::function<Parsable*(Parsable *)> ParsableLoader;
	typedef std::function<std::string (void)> ReferenceGetter;
	typedef std::function<void (std::string *)> ReferenceLoader;
	typedef std::function<void (void*)> VoidCallback;

	class Parsable
	{
	public:
		Parsable();
		virtual ~Parsable();
		virtual void associate() = 0;
		virtual std::string XMLName() const = 0;

	protected:
		void XMLLoad(Beacon *beacon, std::list< std::unordered_map<std::string, ParsableReturn > *> &instanceProvider);
		virtual void postLoading();

		void XMLParse(Beacon *beacon);

		void XMLAssociateField(const std::string &field, XML::Field *xmlfield);

		template<class RDI>
		void XMLAssociateSubBeacon(const std::string &sub, RDI begin, RDI end, ParsableReturn func);
		void XMLAssociateSubBeacon(const std::string &sub, Parsable* element, ParsableReturn func);
		void XMLAssociateSubBeacon(const std::string &sub, Parsable* element);

		template<class RDI>
		void XMLAssociatePolymorph(const std::string &containerName, RDI begin, RDI end, ParsableLoader func);
		void XMLAssociatePolymorph(const std::string &containerName, Parsable *element, ParsableLoader func);

		void XMLAddInstanceProvider(const std::string &type, ParsableReturn provider);

		void XMLAddPostRoutine(const std::string &sub, ParsableReturn routine);

		void XMLAddReference(const std::string &sub, ReferenceGetter getter, ReferenceLoader loader);

	private:

		// For loading
		void getProvider(std::list< std::unordered_map<std::string, ParsableReturn > *> &instanceProvider);
		void loadField(Beacon *beacon);
		void loadBeacon(Beacon *beacon, std::list< std::unordered_map<std::string, ParsableReturn > *> &instanceProvider);
		bool subcreateFunction(Beacon &beacon, std::list< std::unordered_map<std::string, ParsableReturn > *> &instanceProvider);
		bool subElement(Beacon &beacon, std::list< std::unordered_map<std::string, ParsableReturn > *> &instanceProvider);
		bool subSimple(Beacon &beacon, std::list< std::unordered_map<std::string, ParsableReturn > *> &instanceProvider);
		bool subPolymorph(Beacon &beacon, std::list< std::unordered_map<std::string, ParsableReturn > *> &instanceProvider);
		bool subReference(Beacon &beacon);
		void postRoutine(const std::string &sub);

		// For Parsing

		// Members

	protected:
		std::unordered_map<std::string, XML::Field*> *_fieldVariable;
		std::unordered_map<std::string, ParsableReturn > *_subCreateFunction;
		std::unordered_map<std::string, std::vector<Parsable*> > *_subVector;
		std::unordered_map<std::string, Parsable* > *_subElement;
		std::unordered_map<std::string, Parsable* > *_subSimple;
		std::unordered_map<std::string, ParsableReturn > *_postRoutine;

		std::unordered_map<std::string, std::vector<Parsable*> > *_subPolymorphVector;
		std::unordered_map<std::string, ParsableLoader > *_instanceLoader;
		std::unordered_map<std::string, ParsableReturn > *_instanceProvider;

		std::unordered_map<std::string, ReferenceGetter> *_refGetter;
		std::unordered_map<std::string, ReferenceLoader > *_refLoader;

		std::vector<std::string> *_beacon;
		void addBeaconStr(const std::string &str);

		virtual void clearAssociation();
	};

	template<class RDI>
	inline void Parsable::XMLAssociateSubBeacon(const std::string & sub, RDI begin, RDI end, ParsableReturn func)
	{
		if (_subVector == NULL)
			_subVector = new std::unordered_map<std::string, std::vector<Parsable*> >();
		if (_subVector->find(sub) == _subVector->end()) {
			_subVector->insert(std::make_pair(sub, std::vector<Parsable*>()));
		}

		for (auto s = begin; s != end; ++s) {
			(*_subVector)[sub].push_back(*s);
		}
		if (_subCreateFunction == NULL)
			_subCreateFunction = new std::unordered_map<std::string, XML::ParsableReturn >();
		_subCreateFunction->insert(std::make_pair(sub, func));

		addBeaconStr(sub);
	}

	template<class RDI>
	inline void Parsable::XMLAssociatePolymorph(const std::string & containerName, RDI begin, RDI end, ParsableLoader func)
	{
		if (_subPolymorphVector == NULL)
			_subPolymorphVector = new std::unordered_map<std::string, std::vector<Parsable*> >();

		if (_subPolymorphVector->find(containerName) == _subPolymorphVector->end()) {
			_subPolymorphVector->insert(std::make_pair(containerName, std::vector<Parsable*>()));
		}
		for (auto s = begin; s != end; ++s) {
			(*_subPolymorphVector)[containerName].push_back(*s);
		}

		if (_instanceLoader == NULL)
			_instanceLoader = new std::unordered_map<std::string, ParsableLoader >();
		_instanceLoader->insert(std::make_pair(containerName, func));

		addBeaconStr(containerName);
	}

	class DoublePos : public XML::Parsable
	{
	public:
		DoublePos(ge_pd *pos) : _pos(pos) {}
		virtual std::string XMLName() const { return "null"; }
	protected:
		virtual void associate() {
			XMLAssociateField("width", new XML::Double(&(_pos->w)));
			XMLAssociateField("heigh", new XML::Double(&(_pos->h)));
		}
	private:
		ge_pd * _pos;
	};

	class DoubleSquare :
		public XML::Parsable
	{
	public:
		DoubleSquare(square_d *sq) : pos(&(sq->pos)), dim(&(sq->dim)) {}
	protected:
		Parsable * LoadPos(void *v) {
			return &pos;
		}
		Parsable * LoadDim(void *v) {
			return &dim;
		}
		virtual void associate() {
			XMLAssociateSubBeacon("pos", &pos);
			XMLAssociateSubBeacon("dim", &dim);
		}
		virtual std::string XMLName() const { return "proportion"; }

	private:
		XML::DoublePos pos;
		XML::DoublePos dim;
	};

}