#pragma once
#include <XMLCompliant.h>
#include "AnimationDataBase.h"
#include "Surface.h"

#define DEFAULTLETTERLIB "..\\textlib\\mylib.xml"

class LetterLib :
	public XML::Compliant
{
public:
	enum TextSize {VBIG=48, BIG=38, MEDIUM=32, NORMAL=26, SMALL=22, VSMALL=18};
	class Letter : public Animation {
	public:
		Letter(SDL_Texture *texture, const SDL_Rect&dim, const std::string &id);
		void computeRelative(const SDL_Rect & biggest);
		square_d getRelative() const;
		const ge_d &getWidth() const;
		const ge_d &getHeigh() const;
		

		virtual std::string getFilename(void *v);

	private:
		SDL_Rect _dim;
		square_d _prop;
		std::string _id;
	};
public:
	LetterLib();
	~LetterLib();

	void split(const std::string &s, std::vector<std::string > &sentence, std::vector<ge_d > &size);
	virtual std::string XMLName() const;
	static std::string staticXMLName();
	std::string getLibName() const;


	void loadLib(const std::string &file);
	virtual void associate();
	virtual void postLoading();
	const ge_d &getSpaceSize() const;
	void setAsSpace(const char &c);

	bool letterExist(const char &c);
	Letter *getLetter(const char &c);

	static void setRenderer(SDL_Renderer *renderer);
	static LetterLib *getLib(const std::string &name);
	static LetterLib *getDefaultLib();

private:

	void createLetter(const char &c, SDL_Texture *texture, const SDL_Rect &rect);
	std::string _symbolList, _file, _name, _convertSpace;
	ge_d _lineborder, _letterborder;
	SDL_Rect _biggest;
	ge_d _spaceSize;
	std::unordered_map<char, Letter *> _letter;
	std::unordered_map<Letter *, char> _revletter;
	static SDL_Renderer *_renderer;
	static std::unordered_map<std::string, LetterLib*> _lib;
};

