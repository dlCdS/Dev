#include "GraphicalEngine.h"

GE GE::_singleton = GE();

GE::GE() :
	_window("MainWindow"),
	_editable(NULL)
{}

bool GE::Init()
{
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0)
	{
		Log(ERROR, "Échec de l'initialisation de la SDL ", SDL_GetError());
		return false;
	}
	Log(LINFO, "SDL initialized");
	return true;
}

void GE::FreeAll()
{
	AnimationDataBase::freeAll();
	TextureDataBase::freeAll();
}

void GE::SetWindowSize(const ge_pi & size) { _singleton._window.setSize(size); }

bool GE::Start(const ge_i & duration)
{
	_singleton._frameDuration = duration;
	Clocks::start("main");
	_singleton.initLoopback();
	if (!_singleton._window.createWindow()) return false;
	_singleton._windowMap.insert(std::make_pair(_singleton._window.getId(), &_singleton._window));
	_singleton.loadCommonTexture();
	return true;
}

void GE::SaveWindowConfiguration(const std::string & filename)
{
	_singleton._window.save(filename);
}

void GE::LoadWindowConfiguration(const std::string & filename)
{
	_singleton._window.load(filename);
}

void GE::CreateSubWindow(const ge_pi & size)
{
	SubWindow *sub = new SubWindow("sub");
	sub->setSize(size); 
	sub->createWindow();
	_singleton._windowMap.insert(std::make_pair(sub->getId(), sub));
	_singleton._sub.push_back(sub);
}

void GE::DeleteSubWindow(const Uint32 & id)
{
	if (id < 0) {
		for (auto s : _singleton._sub) {
			delete s;
		}
		_singleton._windowMap.clear();
		_singleton._windowMap.insert(std::make_pair(_singleton._window.getId(), &_singleton._window));
	}
	else {
		if (id == _singleton._window.getId())
			Log(ERROR, "Cannot delete main window");
		else {
			auto w = _singleton._windowMap.find(id);
			if (w != _singleton._windowMap.end()) {
				_singleton._sub.remove((SubWindow*)w->second);
				delete w->second;
				_singleton._windowMap.erase(id);
			}
			else Log(ERROR, "No window with id ", id);
		}
	}
}

void GE::SaveEnvironment()
{

}

void GE::ResetWindow()
{
	_singleton._window.freeAll();
}

void GE::Quit()
{
	FreeAll(); SDL_Quit(); DeleteSubWindow();
}

Event * GE::popEvent()
{
	return PublicSdlEventServer::pop();
}

/*
void GE::createWindowWidget()
{
	ListWidget *lst = new ListWidget(&_singleton._window);
	lst->setRelativeProportion(square_d(0.6, 0, 0.4, 1));

	for (int i = 0; i < 2; i++) {
		FitLineWidget *fit = new FitLineWidget(lst);
		ClickWidget *cl = new ClickWidget(fit);
		IconWidget *ic = new IconWidget(cl, AnimationDataBase::requestAnimation(DEFAULTANIMATION));
		ic->setSize(ge_pi(80, 80));
		cl->addWidget(ic);
		fit->addWidget(cl);
		lst->addWidget(fit);
	}
	TextFieldWidget *tf = new TextFieldWidget();
	tf->setTextSize(40 );
	tf->setText("This is Dogodo first text field !");
	lst->addWidget(tf);
	//_singleton._window.addWidget(grid);
	_singleton._window.addWidget(lst);
}
*/

void GE::Step()
{
	_singleton._window.step(); 
	for (auto s : _singleton._sub)
		s->step();
	_singleton.keyCheck();
	Loopback::Solver::SolveEvent();
	_singleton.synchronize();
}

Window * GE::GetWindow(const Uint32 & id)
{
	auto w = _singleton._windowMap.find(id);
	if (w != _singleton._windowMap.end())
		return w->second;
	return nullptr;
}

void GE::AddWidgetToWindow(Widget *w, const Uint32 & id)
{
	Window *win = GetWindow(id);
	if (win != NULL) {
		win->addWidget(w);
	}
	else Log(LERROR, "Failed to add widget to window ", id);
}

void GE::keyCheck() {
	SDL_Event event;
	while (SDL_PollEvent(&event)) {
		switch (event.type) {
		case SDL_WINDOWEVENT:
			_current = GetWindow(event.window.windowID);
			Log(LDEBUG, "Window focus changed");
			break;
		case SDL_MOUSEWHEEL:
			if (event.button.x < 0)
				new MouseScrollEvent(-1);
			else
				new MouseScrollEvent(1);
			break;
		case SDL_KEYDOWN:
			new  KeyPressedEvent(event.key.keysym.sym, true);
			break;
		case SDL_KEYUP:
			new  KeyPressedEvent(event.key.keysym.sym, false);
			break;
		case SDL_MOUSEBUTTONDOWN:
			new MouseClickEvent(event.button.button, true);
			break;
		case SDL_MOUSEBUTTONUP:
			new MouseClickEvent(event.button.button, false);
			break;
		case SDL_MOUSEMOTION:
			new MouseMotionEvent(event.window.windowID, { event.motion.x, event.motion.y });
			break;
		default:
			break;
		}
	}
}

void GE::initLoopback() {
	Loopback::Associate<MouseMotionEvent>(new SolveMouseMotionEvent(&(_current), &_focus));
	Loopback::Associate<MouseClickEvent>(new SolveMouseClickEvent(&_focus));
	Loopback::Associate<EditTextFieldEvent>(new SolveEditTextFieldEvent(&(_editable)));
	Loopback::Associate<KeyPressedEvent>(new SolveKeyPressedEvent(&(_editable)));
	Loopback::Associate<CallbackWidgetRemovedEvent>(new SolveCallbackWidgetRemovedEvent(&_focus));
}

void GE::loadCommonTexture()
{
	AnimationDataBase::addAnimation(COLORED_TRIANGLE, new Animation(TextureDataBase::addTexture(COLORED_TRIANGLE, Surface::getColorTriangle())));
	AnimationDataBase::addAnimation("test", new Animation(TextureDataBase::addTexture("test", Surface::getSurface(_window.getRenderer(), TextureDataBase::requestTexture(COLORED_TRIANGLE)))));
}

void GE::synchronize()
{
	clock_t wait, current;
	static clock_t last(clock());
	if(_frameDuration>=0){
		// Clocks::prepare("main");
		current = clock();
		wait = _frameDuration - 1000 * (current-last) / CLOCKS_PER_SEC;
		//std::cout << wait << std::endl;

		if (wait > 0)
			SDL_Delay(wait);
		else
			Log(LWARNING, " - Frame too long ", wait);
			
		last = clock();
		// Clocks::start("main");
	}
}