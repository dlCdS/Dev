#include "Common.h"

Workspace Workspace::_singleton = Workspace();

std::string Workspace::Path(const std::string & path) {
	if (path == "")
		return _singleton._path;
	else return _singleton._path + "\\" + path;
}

std::string Workspace::GetCurrent()
{
	std::string path("");
#ifdef __COMPILE_WINDOWS__
	char buffer[MAX_PATH];
	GetCurrentDirectoryA(MAX_PATH, buffer);
	path = buffer;
#endif
	return path;
}

std::string Workspace::StartPath()
{
	return _singleton._startPath;
}

std::string Workspace::Explore() {
	return StripFile(GetFile());
}

std::string Workspace::GetFile()
{
	std::string path;
#ifdef __COMPILE_WINDOWS__
	char buffer[MAX_PATH];
	GetCurrentDirectoryA(MAX_PATH, buffer);

	std::cout << buffer << std::endl;

	OPENFILENAMEA ofn;       // common dialog box structure 
	char szFile[260];       // buffer for file name
	HWND hwnd = NULL;              // owner window


							// Initialize OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFile = szFile;
	// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
	// use the contents of szFile to initialize itself.
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "All\0*.*\0Text\0*.TXT\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = (std::string(buffer) + "\\data\\skin").c_str();
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	// Display the Open dialog box. 

	if (GetOpenFileNameA(&ofn) == TRUE)
		path = ofn.lpstrFile;
#endif
	return path;
}

void Workspace::SetPath(const std::string & path)
{
	SetCurrentDirectory(path.c_str());
	_singleton._path = path;
}

void Workspace::Directory(const std::string & path)
{
	CreateDirectory(path.c_str(), NULL);
}

std::string Workspace::StripFile(const std::string & str)
{
	std::string ret(""), cur("");
	for (auto c : str) {
		if (c == '\\'){
			ret += cur + c;
			cur = "";
		}
		else cur += c;
	}
	return ret;
}

Workspace::Workspace() : _startPath(GetCurrent())
{
}
