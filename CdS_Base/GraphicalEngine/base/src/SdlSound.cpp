#include "SdlSound.h"

SdlSound SdlSound::_singleton = SdlSound();

void SdlSound::Init()
{
	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO);
}

int SdlSound::Setup(const ge_i& freq, const Uint16& format, const Uint8& channels, const Uint16& samples)
{
	/*
	if (SDL_OpenAudio(&_singleton._audioSpec, &_singleton._obtained) < 0)
	{
		Log(LERROR, "Failed to setup audio");
		return -1;
	}
	*/
	return -1;
}

int SdlSound::LoadWAV(const std::string& filename)
{
	if (SDL_LoadWAV(filename.c_str(), &_singleton.wav_spec, &_singleton.wav_buffer, &_singleton.wav_length) == NULL)
	{
		Log(LERROR, "Failed to load wav file ", filename);
		return -1;
	}
	Log(LINFO, "Opened wav file of size ", _singleton.wav_length);

	_singleton.wav_spec.callback = AudioCallback;
	_singleton.wav_spec.userdata = NULL;
	// set our global static variables
	_singleton.audio_pos = _singleton.wav_buffer; // copy sound buffer
	_singleton.audio_len = _singleton.wav_length; // copy file length

	if (SDL_OpenAudio(&_singleton.wav_spec, NULL) < 0) {
		Log(LERROR, "Failed to setup audio");
		return -1;
	}

	Log(LINFO, "Opened Wav file with parameters : ", (int)_singleton.wav_spec.channels, " channels, ",
		_singleton.wav_spec.format, " format, ", _singleton.wav_spec.freq, " freq, ",
		_singleton.wav_spec.samples, " samples, ", _singleton.wav_spec.size, " size, ",
		_singleton.wav_spec.padding, " padding");

	return 0;
}

int SdlSound::FreeWAV()
{
	SDL_FreeWAV(_singleton.wav_buffer);
	return 0;
}

void SdlSound::PlayAudio()
{
	if (!_singleton.playing)
		SDL_PauseAudio(0);
}

void SdlSound::WaitAudio()
{
	if (_singleton.end_time == 0) {
		_singleton.end_time = clock();
		while (_singleton.audio_len > 0) {
			SDL_Delay(10);
		}
		Log(LINFO, "Waited Audio ", 1000 * (clock() - _singleton.end_time) / CLOCKS_PER_SEC, " ms (~10)");
	}
	else {
		Log(LINFO, "Audio finished ", 1000 * (clock() - _singleton.end_time) / CLOCKS_PER_SEC, " ms ago");
	}
}

void SdlSound::AudioCallback(void* udata, Uint8* stream, int len)
{
	if (_singleton.audio_len == 0) {
		if (_singleton.end_time == 0){
			_singleton.end_time = clock();
			SDL_PauseAudio(1);
		}
		return;
	}

	len = (len > _singleton.audio_len ? _singleton.audio_len : len);
	SDL_memcpy (stream, _singleton.audio_pos, len); 					// simply copy from one buffer into the other
	//SDL_MixAudio(stream, _singleton.audio_pos, len, SDL_MIX_MAXVOLUME);// mix from one buffer into another

	_singleton.audio_pos += len;
	_singleton.audio_len -= len;
}

SdlSound::SdlSound() : end_time(0), playing(false)
{
}

SdlSound::~SdlSound()
{
	SDL_CloseAudio();
}
