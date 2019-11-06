#pragma once

#include "MainWindow.h"
#include "SubWindow.h"
#include <Clocks.h>
#include "LoopbackAction.h"
#


#define DEFAULT_FRAME_DURATION 1 // ms


class GE  {
		public:

			static bool Init();
			static void FreeAll();
			static void SetWindowSize(const ge_pi &size = DEFAULT_WINDOW_SIZE);
			static bool Start(const ge_i &duration = DEFAULT_FRAME_DURATION);
			static void SaveWindowConfiguration(const std::string &filename);
			static void LoadWindowConfiguration(const std::string &filename);

			static void CreateSubWindow(const ge_pi &size);
			static void DeleteSubWindow(const Uint32 &id=-1);
			
			static void SaveEnvironment();

			static void ResetWindow();
			static void Quit();
			static Event* popEvent();
			static void Step();
			static Window* GetWindow(const Uint32 &id);
			static void AddWidgetToWindow(Widget *w, const Uint32 &id = _singleton._window.getId());

	private:
		GE();

		void keyCheck();

		void initLoopback();
		void loadCommonTexture();

		std::list<Widget*> _focus;
		MainWindow _window;
		Window *_current;
		std::list<SubWindow*> _sub;
		std::unordered_map<Uint32, Window*> _windowMap;
		ge_i _frameDuration; //ms

		static GE _singleton;

		TextFieldWidget *_editable;

		void synchronize();
};