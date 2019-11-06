/* DOCUMENTATION

The XML Parsing class for which XMLCompliant is the main class
are a set inheritable pattern to make objects loadable and savable
in XML format. To make a class XML compatible, you can:

* Declare it XMLCompliant 
	if you want to be able to save its structure in an XML file
	
* Declare it XMLParsable
	if it is just needs to be loaded by a XMLCompliant

* The XMLParser class provide a datatype storing XML content

To make a class inherite from XMLParser, there are five mandatory steps
1)
	Overload the "virtual std::string XMLName() const" function so it
	returns the class XML name
	A good approach is to create a "static std::string XMLName()" function
	returning the class XML name. Then you'll be able to reuse this function
	to make XML associations

2)
	Overload the "virtual void associate()" function in order to make the
	boundage between the class members and the XML Fields and Beacon. This
	function may contain six potential functions to link your inner variables

3)
	The previously introduced associate method can contain function link to several types of datastructures:

	* Common types (int, double std::string... ) #Common types container TBD
	* XMLParsable inheriting objects or containers of XMLParsable inheriting objects
	* XMLParsable polymophic objects or containers of XMLParsable polymophic objects

	The latest type will be described latter

4)
	Common types are considered as Fields. To associate a simple field, use the function
	> void XMLAssociateField(const std::string &field, XML::Field *xmlfield)

	Let say you have a "int myInteger" in your class, then put the function
	> XMLAssociateField("myInt", &myInteger);
	To the "associate" funtion

5)
	To add a simple element with inherite from XMLParsable class, use the function
	> void XMLAssociateSubBeacon(const std::string &sub, Parsable* element)

	This objet will be loaded thanks to it's own associations
	If you need to process this object in a particular way at intanciation, use the function
	> void XMLAssociateSubBeacon(const std::string &sub, Parsable* element, ParsableReturn func)

	This function allows you to specify which function creates the objet. The
	> ParsableReturn func
	Must be replaced by a function of type
	> XML::XMLParsable *function(void *v)
	No argument is passed to this function, you can skip the argument
	> void *v
	This function must return the instanciated object so it can be loaded

	If you need to load a container of XMLParsable objects, then use the function
	> template<class RDI>
	> void XMLAssociateSubBeacon(const std::string &sub, RDI begin, RDI end, ParsableReturn func)
	Where RDI is the iterator bounds of the container

	Information about
	> ParsableReturn
	This type is a typedef on
	> std::function<Parsable*(void *v)>
	It can be easily got by using the macro
	> SubBeaconLoadFunction

	Let say you need to print something when instaciating the objet from class MyClass
	> MyObject *object

	First create a function

	> XMLParsable *MyClass::myObjectIntanciator(void *v) {
	>	std::cout << "Loading my Object" << std::endl;
	>	object = new MyObject();
	>	return object;
	> }

	Then add the following line in the associate function
	> XMLAssociateSubBeacon("object", object, SubBeaconLoadFunction(MyClass, myObjectIntanciator))

	Note that if you do not want the newly created objec to be loaded, you can return NULL

6)
	If your class uses polymorphic objects, then you need to give an instanciator to the load function
	Let say you have

	> class Base {};
	> class Derived[0, 1, 2, ...] : public Base {};
	> class User {
	>	Base *base;
	>	}
	> class Main {
	>	User user;
	> }

	When loading the "base" member of "user", User class may not have the required information to instanciate
	the derived class. To make possible this polymorphism, create a loader function in the User class

	> XMLParsable *User::loadBase(XMLParsable *instance) {
	>	base = dynamic_cast<Base *>(instance);
	>	return base;
	> }

	Then associate it in the association

	void XMLAssociateSubBeacon(const std::string &sub, RDI begin, RDI end, ParsableReturn func);

		void XMLAssociateSubBeacon(const std::string &containerName, const std::string &baseClassName, ParsableLoader func);

		void XMLAddInstanceProvider(const std::string &type, ParsableReturn provider);


*/





#pragma once
#include "XMLParsable.h"

namespace XML {
	class Compliant;
	class Base : public Parsable {
	public:
		Base(Parsable *parent, const std::string &baseName);
		void parse(Beacon *b);
		void load(Beacon *b);
		virtual std::string XMLName() const { return "XMLBase"; }
		const std::string &getName() const;
	protected:
		virtual void associate();
		Parsable *add(void *v);
		Parsable *_parent;
		std::string _baseName;
	};

	class Compliant :
		public virtual Parsable
	{
	public:
		Compliant(const std::string &baseName);
		~Compliant();

		void load(Parser *parser);
		void load(const std::string &filename);
		void save(const std::string &filename);

	protected:
		Base _base;
	};

}