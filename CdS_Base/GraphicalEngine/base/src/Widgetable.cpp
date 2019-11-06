#include "Widgetable.h"

#define WIDGETABLE_HEADER_LIST "WIDGETABLE_HEADER_LIST"

Widgetable::Widgetable(const std::string &name) : XML::Compliant(name),
_editable(true),
_displayHeader(true),
_target(&_container),
_prevPage(NULL),
_main(NULL),
_isShared(false),
_callback(NULL),
_display(NULL),
_order(NULL),
_associate(NULL),
_displayField(NULL),
_displayForward(NULL),
_updateVar(NULL),
_widgetablePage(NULL),
_building(false)
{
	_container.setShared();
}

Widgetable::Widgetable(const std::string & name, const bool & shared) : XML::Compliant(name),
_editable(true),
_displayHeader(true),
_target(&_container),
_prevPage(NULL),
_main(NULL),
_isShared(shared),
_callback(NULL),
_display(NULL),
_order(NULL),
_associate(NULL),
_displayField(NULL),
_displayForward(NULL),
_updateVar(NULL),
_widgetablePage(NULL),
_building(false)
{
	_container.setShared();
}



Widgetable::~Widgetable()
{
	_container.freeAll();
	if(_widgetablePage != NULL)
		for (auto w : *_widgetablePage)
			if (!w.second->_isShared)
				delete w.second;
}

Widget * Widgetable::getDetachableContainer()
{
	_target = new PageWidget();
	ListWidget *main = _main;
	_main = new ListWidget();
	_target->addWidget(_main, "Main");
	build();
	_main = main;
	PageWidget *ret = _target;
	_target = &_container;
	return ret;
}

Widget * Widgetable::getContainer()
{
	return &_container;
}

void Widgetable::setAsPage(Widgetable *prev)
{
	_prevPage = prev;
}

void Widgetable::WAddCallback(const std::string & name, XML::VoidCallback func)
{
	if(_building){
		if (_callback == NULL)
			_callback = new std::unordered_map<std::string, XML::VoidCallback>();
		_callback->insert(std::make_pair(name, func));
		addBeaconStr(name);
	}
}

void Widgetable::WAssociateWidget(const std::string & widpos, Widget * wid)
{
	if (_building){
		if (_display == NULL)
			_display = new std::vector<BuildWidget>();
		_display->push_back(BuildWidget(widpos, wid));
	}
	else if (!wid->isShared()) {
		delete wid;
	}
}

void Widgetable::WAssociateBeacon(const std::string & item, const std::string & widpos)
{
	if (_building){
		if (_associate == NULL)
			_associate = new std::unordered_map<std::string, std::string>();
		if (_order == NULL)
			_order = new std::vector<std::string>();
		_associate->insert(std::make_pair(item, widpos));
		_order->push_back(item);
	}
}

void Widgetable::WAssociateField(const std::string & item, const std::string & widpos, XML::Field * field)
{
	if (_building){
		if (_displayField == NULL)
			_displayField = new std::unordered_map<std::string, XML::Field *>();
		WAssociateBeacon(item, widpos);
		XMLAssociateField(item, field->getCopy());
		_displayField->insert(std::make_pair(item, field));
	}
	else {
		XMLAssociateField(item, field);
	}
}

void Widgetable::WAssociateField(const std::string & item, const std::string & widpos, XML::Field * field, XML::VoidCallback forwardChange)
{
	if (_building){
		if (_displayForward == NULL)
			_displayForward = new std::unordered_map<std::string, XML::VoidCallback>();
		WAssociateField(item, widpos, field);
		_displayForward->insert(std::make_pair(item, forwardChange));
	}
	else {
		XMLAssociateField(item, field);
	}
}

void Widgetable::pageBack(void * v)
{
	_prevPage->backMain();
}

void Widgetable::selectWidget( void * v)
{
	_target->setPage(*(std::string*)v);
}

bool Widgetable::createField(const std::string & item, Widget * dest)
{
	if (_displayField != NULL && _displayField->find(item) != _displayField->end()) {
		XML::Field *f = (*_displayField)[item];
		FieldWidget *fw = new FieldWidget();
		fw->setTextSize(LetterLib::SMALL);
		fw->associateField(item, f);
		if (_displayForward != NULL && _displayForward->find(item) != _displayForward->end())
			fw->setForwardFunction((*_displayForward)[item]);
		dest->addWidget(fw);
		addUpdateVar(f->getVar(), fw);
		return true;
	}
	else return false;
}

bool Widgetable::createPage(const std::string &item, Widget * dest)
{
	if (_widgetablePage != NULL && _widgetablePage->find(item) != _widgetablePage->end()) {
		Widgetable *wab = (*_widgetablePage)[item];
		wab->setAsPage(this);
		wab->build();
		_target->addWidget(wab->getContainer(), item);

		ClickWidget *cw = new ClickWidget();
		cw->setText(item);
		cw->setCallback(VoidedCallbackFunction(Widgetable, selectWidget), true);
		cw->useTextAsParameter();
		dest->addWidget(cw);
		return true;
	}
	else return false;
}

void Widgetable::resetContainer()
{
	_container.freeAll();
	_main = new ListWidget();
	_container.addWidget(_main, MAIN_PAGE_NAME);
}

Widget * Widgetable::getFromName(const std::string & name, std::unordered_map<std::string, Widget*> &_fromName)
{
	std::string id(""), cur("");
	bool first(true);
	for (auto c : name) {
		if (c == ':') {
			if (!first) 
				id += c;
			else first = false;
			id += cur;
			cur = "";
		}
		else
			cur += c;
	}
	return _fromName[id];
}

void Widgetable::buildWidget(std::unordered_map<std::string, Widget*> &_fromName)
{
	_fromName.insert(std::make_pair("", _main));
	if(_display != NULL){
		for (auto bw : *_display) {
			_fromName.insert(std::make_pair(bw._widpos, bw._w));
			getFromName(bw._widpos, _fromName)->addWidget(bw._w);
		}
	}
}

void Widgetable::addUpdateVar(void * var, Widget *w)
{
	if (_updateVar == NULL)
		_updateVar = new std::unordered_map<void *, Widget *>();
	_updateVar->insert(std::make_pair(var, w));
}

void Widgetable::build()
{
	_building = true;
	std::unordered_map<std::string, Widget*> _fromName;
	resetContainer();
	if (_displayHeader) {
		_main->addWidget(new TextFieldWidget(LetterLib::NORMAL, _base.getName()));
	}
	if(_prevPage != NULL){
		ClickWidget *cw = new ClickWidget();
		cw->setText("Return");
		cw->setCallback(VoidedCallbackFunction(Widgetable, pageBack), true);
		_main->addWidget(cw);
	}
	associate();
	buildWidget(_fromName);

	std::string be;
	if(_order != NULL)
	for (auto be : *_order) {
		auto ass = (*_associate)[be];
		auto wid = _fromName.find(ass);
		if(wid != _fromName.end()) {
			if (createField(be, wid->second)) ;
			else if (createPage(be, wid->second));
			if (_subSimple  != NULL && _subSimple->find(be) != _subSimple->end()) {
				Widgetable *sub = dynamic_cast<Widgetable*>((*_subSimple)[be]);
				sub->build();
				wid->second->addWidget(sub->getContainer());
			}
			if (_subElement  != NULL && _subElement->find(be) != _subElement->end()) {
			}
			if (_subVector  != NULL && _subVector->find(be) != _subVector->end()) {
				for (auto s : (*_subVector)[be]) {
				}
			}
			if (_subPolymorphVector  != NULL && _subPolymorphVector->find(be) != _subPolymorphVector->end()) {
			}
			if (_refGetter != NULL && _refGetter->find(be) != _refGetter->end()) {
			}
			if (_callback != NULL && _callback->find(be) != _callback->end()) {
				ClickWidget *cw = new ClickWidget();
				cw->setText(be);
				cw->setCallback((*_callback)[be], true);
				wid->second->addWidget(cw);
			}
		}
	}
	clearAssociation();
	_building = false;
}

void Widgetable::variableChanged(void * var)
{
	if(_updateVar != NULL) {
		auto it = _updateVar->find(var);
		if (it != _updateVar->end())
			it->second->updateContent();
	}
}

void Widgetable::WAddPage(const std::string & name, const std::string &widpos, Widgetable * widget)
{
	if (_building){
		if (_widgetablePage == NULL)
			_widgetablePage = new std::unordered_map<std::string, Widgetable *>();
		WAssociateBeacon(name, widpos);
		_widgetablePage->insert(std::make_pair(name, widget));
	}
}

void Widgetable::backMain()
{
	_target->setPage(MAIN_PAGE_NAME);
}

void Widgetable::deleteContainer(void * v)
{
	if (v != NULL) {
		delete v;
		v = NULL;
	}
}

void Widgetable::clearAssociation()
{
	XML::Compliant::clearAssociation();
	deleteContainer(_callback);
	deleteContainer(_display);
	deleteContainer(_order);
	deleteContainer(_associate);
	deleteContainer(_displayForward);
	deleteContainer(_displayField);
	_callback = NULL;
	_display = NULL;
	_order = NULL;
	_associate = NULL;
	_displayForward = NULL;
	_displayField = NULL;
}
