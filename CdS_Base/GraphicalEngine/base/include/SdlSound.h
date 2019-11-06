#pragma once
#include <SDL.h>
#include <Common.h>


class SdlSound
{
public:
	static void Init();

	static int Setup(const ge_i& freq, const Uint16& format, const Uint8& channels, const Uint16& samples);

	static int LoadWAV(const std::string& filename);
	static int FreeWAV();
	static void PlayAudio();
	static void WaitAudio();

private:
	static void AudioCallback(void* udata, Uint8* stream, int len);
	SdlSound();
	~SdlSound();

	static SdlSound _singleton;
	Uint8* audio_pos; // global pointer to the audio buffer to be played
	Uint32 audio_len; // remaining length of the sample we have to play

	Uint32 wav_length; // length of our sample
	Uint8* wav_buffer; // buffer containing our audio file
	SDL_AudioSpec wav_spec; // the specs of our piece of music
	clock_t end_time;
	bool playing;
};

