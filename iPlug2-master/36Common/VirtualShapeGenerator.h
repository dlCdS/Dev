#pragma once
#include "IPlugMidi.h"
#include <mutex>
#include <list>
#include "Math36.h"

using namespace iplug;

/* Possible improvment:
* have a pitched attack - done
* linking modwheels
* have a filtered attack
* add lfo gain & lfo pitch
* ?
*/

class VirtualShapeGenerator
{
  struct Note;
  struct SimpleMidi;
public:
  VirtualShapeGenerator();

  void ProcessMidiMsg(const IMidiMsg &msg);
  void ProcessBlock(sample** outputs, int nChannel, int nFrames);

  void Reset(const double& sampleRate);
  void EnableGlide(const bool& enable);
  void SetAttack(const double& attackValue);
  void SetSustain(const double& sustainValue);
  void SetDecay(const double& decayValue);
  void SetRelease(const double& releaseValue);
  void SetGlideTime(const double& glideTimeValue);
  void SetGain(const double& gainValue);
  void SetVoices(const int& voicesValue);
  void SetSpread(const double& spreadValue);
  void EnablePitch(const bool& enable);
  void SetPitchAttack(const double& pitch);
  void SetPitchTime(const double& pitchtime);
  void SetPitchBend(const double& pitchbend);

protected:
  /// <summary>
  /// Function to return normalized shape value in an interval of [-1, 1]
  /// Make sure the shapes makes one period in the interval [0, 1]
  /// ///<param name="paramName">t</param>: The corresponding time between [0, 1] 
  /// </summary>
  virtual sample getShape(const double& t) = 0;

private:
  std::list<int> _noteOn, _noteOff;
  bool _enableGlide, _enablePitch;
  double _attack, _sustain, _decay, _release, _glideTime, _gain, _spread, _pitchAttack, _pitchTime, _pitchBend;
  int _voices;
  double _sampleRate, _sampleDuration;

  std::mutex _noteOnMutex, _noteOffMutex;
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
    int _note, _pitch;
    void fromDouble(const double& value);
    double toDouble();
    void operator +=(const SimpleMidi& msg);
    SimpleMidi(const IMidiMsg& msg);
    SimpleMidi(const SimpleMidi& msg);
    SimpleMidi();
  };

  struct Note {
    SimpleMidi _current, _target;
    double _startNote, _endNote;
    double _period[2];
    double _velocity;
    double _time, _glideElapsed, _periodTime[2], _releaseElapsed, _pitchElapsed;
    double _lastGain;
    double _spread;
    bool _glide, _pitch;
    bool _isPlaying, _isStopping;
    Note();
    Note(const IMidiMsg& msg, const double& spreadValue);
    void reset(const IMidiMsg& msg, const double& spreadValue);
    void start();
    void stop();
    double normalizedPeriodLocation(const int &channel = 0) const;
    void forceStop();
    void doGlideFrom(const Note& note);
    void doPitchAttack(const double &pitch);
    void increment(const double& timestep, const double& glideTime, const double & pitchTime, const double& spreadValue);
    bool legato(const double& elapsed, const double& duration);
  };


  Note _noteTab[128];
};
