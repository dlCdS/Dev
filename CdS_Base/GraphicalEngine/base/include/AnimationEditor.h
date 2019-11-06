#pragma once
#include "Widgetable.h"
#include "AnimationDataBase.h"

#define ANIMATION_EDITOR_PAGE_NAME "AnimationEditorPage"
class AnimationEditor :
	public Widgetable
{
public:
	class EditorColor : public Widgetable {
	public:
		EditorColor(const std::string &name);
		~EditorColor();


		std::string XMLName() const;
		virtual void associate();
		void colorChanged(const Color &color);
		Color getColor() const;
	private:

		void satuColorChanged(void *v);
		void colorTextChanged(void *v);
		void satuTextChanged(void *v);
		GetColorWidget *_satu;
		Color *_color;
		ColorS _satuCol;
	};

	~AnimationEditor();

	void newAnimation(void *v);
	void loadAnimation(void *v);


	static void Build();
	static Widget *GetContainer();
	static Widgetable *GetSingleton();
	static void SetSavePath(const std::string &path);
	static void SetAnimation(Animation *animation);

	std::string XMLName() const;
	virtual void associate();

private:
	AnimationEditor();
	void colorChanged(void *v);
	Animation *_anim;
	std::vector<SDL_Surface *> _surfaces;

	std::string _savePath;
	EditorColor _right, _left;
	GetColorWidget *_colorWid;
	SurfaceEditorWidget *_editor;

	static AnimationEditor _sing;
};

