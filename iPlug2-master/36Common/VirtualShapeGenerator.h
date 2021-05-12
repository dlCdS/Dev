#pragma once
#include <mutex>
#include <list>

#include "IPlugMidi.h"
#include "IPlugProcessor.h"
#include "Math36.h"

using namespace iplug;
const int FFT_Size = 2048;

/* Possible improvment:
* have a pitched attack - done
* linking modwheels
* have a filtered attack
* add lfo gain & lfo pitch
* ?
*/

class VirtualShapeGenerator
{
public:
  struct Note;
  struct SimpleMidi;
  enum LaunchMode {
      ONCE,
      ALWAYS
  };

  VirtualShapeGenerator();

  void ProcessMidiMsg(const IMidiMsg &msg);
  void ProcessBlock(sample** outputs, const int& nChannel, const int& nFrames, const int& samplePos, const double& samplesPerBeat, const double& sampleRate);

  void Reset(const double& sampleRate);

  ///  GLIDE
  void EnableGlide(const bool& enable);
  void SetGlideTime(const double& glideTimeValue);
  void SetVoices(const int& voicesValue);

  /// ENVELOPE
  void SetAttack(const double& attackValue);
  void SetSustain(const double& sustainValue);
  void SetDecay(const double& decayValue);
  void SetRelease(const double& releaseValue);

  double GetEnvelope(const double& x);

  /// GAIN
  void SetGain(const double& gainValue);

  /// MAIN
  void SetSpread(const double& spreadValue);
  void SetDetune(const double& detune);
  void SetPitch(const int& pitch);
  void SetPhase(const double& phase);

  /// PITCH ATTACK
  void EnablePitch(const bool& enable);
  void SetPitchAttack(const double& pitch);
  void SetPitchTime(const double& pitchtime);


  /// MOD WHEEL
  void SetPitchBend(const double& pitchbend);
  void SetPitchRange(const double& pitchrange);


  /// FFT STUFF
  double GetPreEffectFFT(const double& freq);
  double GetPostEffectFFT(const double& freq);


  /// ATTACK FILTER
  void EnableAttackFilter(const bool& enable);
  void SetAttackFilterType(const Math36::Filter::FilterMode& type);
  void SetAttackFilterTrig(const LaunchMode& type);
  void SetAttackFilterFromF(const double& fromf);
  void SetAttackFilterFromN(const double& fromn);
  void SetAttackFilterToF(const double& tof);
  void SetAttackFilterToN(const double& ton);
  void SetAttackFilterAtt(const int& att);
  void SetAttackFilterFollow(const bool& follow);
  void SetAttackFilterResFrom(const double& resfrom);
  void SetAttackFilterResTo(const double& resto);
  void SetAttackFilterTime(const double& time);

  double GetAttackFilter(const double& freq);

  /// LFO FILTER
  void SetLFOFilterType(const Math36::Filter::FilterMode& type);
  void SetLFOFilterMix(const double& mix);
  double GetLFOFilter(const double& freq);

  /// PUBLIC LFO
  Math36::LFO _lfopitch, _lfogain, _lfofilter;

protected:
  /// <summary>
  /// Function to return normalized shape value in an interval of [-1, 1]
  /// Make sure the shapes makes one period in the interval [0, 1]
  /// ///<param name="paramName">t</param>: The corresponding time between [0, 1] 
  /// </summary>
  virtual sample getShape(const double& t) = 0;

private:
  virtual sample getShapeWithPhase(const double& t, const double& phase);
  std::list<int> _noteOn, _noteOff;
  bool _enableGlide, _enablePitch;
  double _attack, _sustain, _decay, _release, _glideTime, _gain, _spread, _pitchAttack, _pitchTime, _pitchBend, _pitchRange, _globalPitch, _detune, _phase;
  int _voices, _pitch;
  double _sampleRate, _sampleDuration;

  std::mutex _noteOnMutex, _noteOffMutex;
  void startNote(const IMidiMsg& msg);
  void stopNote(const IMidiMsg& msg);
  // static stuff

  bool _lfopitchnotetrig, _lfogainnotetrig, _lfofilternotetrig;
  bool _lfopitchnotetrigonce, _lfogainnotetrigonce, _lfofilternotetrigonce;

  Math36::FFT fftPreEffect, fftPostEffect;
  Math36::Filter _lfoFilter[2], _attackFilter[2];
  double _lfoFilterMix;
  bool _enableAttackFilter, _fa_follow;
  double _fa_fromf, _fa_tof, _fa_fromn, _fa_ton, _fa_resfrom, _fa_resto, _fa_time, _fa_time_count;
  LaunchMode _fa_launch;
  void filterAttackIncrement(const double& t);
  void filterNewNote();

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
    void setSpread(const double& spreadValue);
    void start();
    void stop();
    double normalizedPeriodLocation(const int &channel = 0) const;
    void forceStop();
    void doGlideFrom(const Note& note);
    void doPitchAttack(const double &pitch);
    void increment(const double& timestep, const double& glideTime, const double & pitchTime);
    bool legato(const double& elapsed, const double& duration);
  };


  Note _noteTab[128];
};
