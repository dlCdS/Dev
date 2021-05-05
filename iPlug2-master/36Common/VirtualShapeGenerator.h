#pragma once
#include "IPlugInstrument.h"
#include <mutex>
#include <list>



class VirtualShapeGenerator
{
  struct Note;
  struct SimpleMidi;
public:
  VirtualShapeGenerator();

  void ProcessMidiMsg(const IMidiMsg &msg);
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames, int sampleRate);

  /// <summary>
  /// Function to return normalized shape value in an interval of [-1, 1]
  /// Make sure the shapes makes one period in the interval [0, 1]
  /// ///<param name="paramName">t</param>: The corresponding time between [0, 1] 
  /// </summary>
  virtual double getShape(const double& t) = 0;
  
  std::list<Note> noteOn, noteOff;
  bool mono;
  double attack, sustain, decay, release, glide;

  std::mutex noteMutex;

  // static stuff

  static double GetFreqFromMidi(const IMidiMsg& msg);
  static double GetFreqFromNote(const int& note, const int& pitch);
  static double GetFreqFromNote(const SimpleMidi& simple);

  static double s_noteToFreq[128][8192];
  static bool s_init;
  static std::mutex s_mutex;
  static void InitStaticStuff();

  struct SimpleMidi {
    int note, pitch;
    void fromDouble(const double& value);
    double toDouble();
    void operator +=(const SimpleMidi& msg);
    SimpleMidi(const IMidiMsg& msg);
    SimpleMidi();
  };

  struct Note {
    SimpleMidi current, target;
    double freq;
    double velocity;
    double time;
    bool glide;
    double glidepos;
    Note();
    Note(const IMidiMsg& msg);
    void doGlide(const IMidiMsg& msg);
  };
};
