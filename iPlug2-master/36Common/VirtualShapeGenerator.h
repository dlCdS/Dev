#pragma once
#include "IPlugMidi.h"
#include <mutex>
#include <list>

using namespace iplug;

class VirtualShapeGenerator
{
  struct Note;
  struct SimpleMidi;
public:
  VirtualShapeGenerator();    

  void ProcessMidiMsg(const IMidiMsg &msg);
  void ProcessBlock(sample** outputs, int nChannel, int nFrames);

  void Reset(const double& sampleRate);
  void SetGlide(const bool& glideValue);
  void SetAttack(const double& attackValue);
  void SetSustain(const double& sustainValue);
  void SetDecay(const double& decayValue);
  void SetRelease(const double& releaseValue);
  void SetGlideTime(const double& glideTimeValue);
  void SetGain(const double& gainValue);
  void SetVoices(const int& voicesValue);

protected:
  /// <summary>
  /// Function to return normalized shape value in an interval of [-1, 1]
  /// Make sure the shapes makes one period in the interval [0, 1]
  /// ///<param name="paramName">t</param>: The corresponding time between [0, 1] 
  /// </summary>
  virtual sample getShape(const double& t) = 0;

private:
  std::list<int> noteOn, noteOff;
  bool enableGlide;
  double attack, sustain, decay, release, glideTime, gain;
  int voices;
  double sampleRate, sampleDuration;

  std::mutex noteOnMutex, noteOffMutex;
  void startNote(const IMidiMsg& msg);
  void stopNote(const IMidiMsg& msg);
  // static stuff

  static double GetDurationFromMidi(const IMidiMsg& msg);
  static double GetDurationFromNote(const int& note, const int& pitch);
  static double GetDurationFromNote(const SimpleMidi& simple);

  static double s_noteToDuration[128][8192];
  static bool s_init;
  static std::mutex s_mutex;
  static void InitStaticStuff();

  // structs
  struct SimpleMidi {
    int note, pitch;
    void fromDouble(const double& value);
    double toDouble();
    void operator +=(const SimpleMidi& msg);
    SimpleMidi(const IMidiMsg& msg);
    SimpleMidi(const SimpleMidi& msg);
    SimpleMidi();
  };

  struct Note {
    SimpleMidi current, target;
    double startNote, endNote;
    double period;
    double velocity;
    double time, glidetime, periodTime, releaseTime;
    double lastGain;
    bool glide;
    bool isPlaying, isStopping;
    Note();
    Note(const IMidiMsg& msg);
    void reset(const IMidiMsg& msg);
    void start();
    void stop();
    double normalizedPeriodLocation() const;
    void forceStop();
    void doGlideFrom(const IMidiMsg& msg);
    void doGlideFrom(const Note& note);
    void increment(const double& timestep, const double& glideTime);
  };


  Note noteTab[128];
};
