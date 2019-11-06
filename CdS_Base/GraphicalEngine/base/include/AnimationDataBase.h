#pragma once
#include "Animation.h"
class AnimationDataBase
{
public:
	~AnimationDataBase();

	static Animation *requestAnimation(const std::string &filename, const std::string &newname="");
	static bool animationExist(const std::string &filename);
	static Animation *getDefaultAnimation();
	static std::string getKey(const Animation *animation);
	static void freeAll();
	static void addAnimation(const std::string &name, Animation *anim);

	static void updateAnimations();

private:
	std::unordered_map<std::string, Animation*> _data;

	AnimationDataBase();
	static AnimationDataBase _singleton;
};

