#include "Wav.h"

WavFile::WavFileData::WavFileData(const char* name) : f(name, ios::binary), size(0), data_chunk_pos(0)
{
}
