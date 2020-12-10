#include <cmath>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;

#define SUBCHUNK1SIZE   (16)
#define AUDIO_FORMAT    (1) /*For PCM*/
#define NUM_CHANNELS    (1)
#define SAMPLE_RATE     (96000)

#define BITS_PER_SAMPLE (32)

#define BYTE_RATE       (SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE/8)
#define BLOCK_ALIGN     (NUM_CHANNELS * BITS_PER_SAMPLE/8)


namespace little_endian_io
{
  template <typename Word>
  std::ostream& write_word(std::ostream& outs, Word value, unsigned size = sizeof(Word))
  {
    for (; size; --size, value >>= 8)
      outs.put(static_cast <char> (value & 0xFF));
    return outs;
  }
}

using namespace little_endian_io;
namespace WavFile {
  struct WavFileData {
    WavFileData(const char* name);
    size_t data_chunk_pos;
    ofstream f;
    size_t size;
  };

  inline WavFileData* initFile(const char* name) {
    WavFileData* data = new WavFileData(name);

    // Write the file headers
    data->f << "RIFF----WAVEfmt ";     // (chunk size to be filled in later)
    write_word(data->f, SUBCHUNK1SIZE, 4);  // no extension data
    write_word(data->f, AUDIO_FORMAT, 2);  // PCM - integer samples
    write_word(data->f, NUM_CHANNELS, 2);  // two channels (stereo file)
    write_word(data->f, SAMPLE_RATE, 4);  // samples per second (Hz)
    write_word(data->f, BYTE_RATE, 4);  // (Sample Rate * BitsPerSample * Channels) / 8
    write_word(data->f, BLOCK_ALIGN, 2);  // data block size (size of two integer samples, one for each channel, in bytes)
    write_word(data->f, BITS_PER_SAMPLE, 2);  // number of bits per sample (use a multiple of 8)

    // Write the data chunk header
    data->data_chunk_pos = data->f.tellp();
    data->f << "data----";  // (chunk size to be filled in later)
    return data;
  }

  inline int closeFile(WavFileData* data) {

    // (We'll need the final file size to fix the chunk sizes above)
    size_t file_length = data->f.tellp();

    int32_t subchunk2_size = data->size * NUM_CHANNELS * BITS_PER_SAMPLE / 8;
    int32_t chunk_size = chunk_size = 4 + (8 + SUBCHUNK1SIZE) + (8 + subchunk2_size);;

    // Fix the data chunk header to contain the data size
    data->f.seekp(data->data_chunk_pos + 4);
    write_word(data->f, subchunk2_size, 4);

    // Fix the file header to contain the proper RIFF chunk size, which is (file size - 8) bytes
    data->f.seekp(0 + 4);
    write_word(data->f, chunk_size, 4);
    data->f.close();
    delete data;
    return 0;
  }

  template <typename Sample>
  inline void printSignal(WavFileData* data, const Sample sample)
  {

    // Write the audio samples
    // (We'll generate a single C4 note with a sine wave, fading from left to right)

    const double max_amplitude = pow(2.0, double(BITS_PER_SAMPLE - 1)) - 8;  // "volume"

    write_word(data->f, (int)(max_amplitude * sample));
    data->size++;
  }

}