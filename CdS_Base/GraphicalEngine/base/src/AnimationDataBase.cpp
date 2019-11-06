#include "AnimationDataBase.h"

AnimationDataBase AnimationDataBase::_singleton = AnimationDataBase();

AnimationDataBase::AnimationDataBase() 
{
}


AnimationDataBase::~AnimationDataBase()
{
	freeAll();
}

Animation * AnimationDataBase::requestAnimation(const std::string & filename, const std::string &newname)
{
	if (_singleton._data.find(filename) != _singleton._data.end()) {
		
		if(_singleton._data[filename]!=NULL)
			return _singleton._data[filename];
		else 
			getDefaultAnimation();
	}
	else if (newname != "" && _singleton._data.find(newname) != _singleton._data.end()) {

		if (_singleton._data[newname] != NULL)
			return _singleton._data[newname];
		else
			getDefaultAnimation();
	}

	std::fstream f(filename, std::ios::in);
	if (!f) {
		std::cout << "unable to load animation " << filename << std::endl;
		_singleton._data.insert(std::make_pair(filename, (Animation*)NULL));
		return getDefaultAnimation();
	}
	f.close();

	std::string used("");
	if (newname != "")
		used = newname;
	Animation *animation = new Animation();
		animation->load(filename.c_str(), used);
		_singleton._data.insert(std::make_pair(used, animation));
	return animation;
}

bool AnimationDataBase::animationExist(const std::string & filename)
{
	if (_singleton._data.find(filename) != _singleton._data.end())
		return true;
	return false;
}

Animation * AnimationDataBase::getDefaultAnimation()
{
	std::string name = DEFAULT_TEXTURE_NAME;

	Animation *animation = requestAnimation(DEFAULTANIMATION);
	if (animation != NULL)
		return animation;
	if (_singleton._data.find(name) != _singleton._data.end())
		return _singleton._data[name];

	animation = Animation::getDefaultAnimation();
	_singleton._data.insert(std::make_pair(name, animation));
	return animation;
}

std::string AnimationDataBase::getKey(const Animation * animation)
{
	auto name = std::find_if(
		_singleton._data.begin(),
		_singleton._data.end(),
		[&animation](const  std::unordered_map<std::string, Animation*>::value_type& vt)
	{ return vt.second == animation; });

	if (_singleton._data.end() != name)
		return name->first;
	else
		return "NULL";
}

void AnimationDataBase::freeAll()
{
	for (auto t = _singleton._data.begin(); t != _singleton._data.end(); ++t)
		delete t->second;
	_singleton._data.clear();
}

void AnimationDataBase::addAnimation(const std::string & name, Animation * anim)
{
	if (_singleton._data.find(name) != _singleton._data.end()) {
		delete _singleton._data[name];
		_singleton._data[name] = anim;
	}
	else {
		_singleton._data.insert(std::make_pair(name, anim));
	}
}

void AnimationDataBase::updateAnimations()
{
	Animation::updateTime();
}
