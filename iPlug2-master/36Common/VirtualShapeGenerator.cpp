#include "VirtualShapeGenerator.h"

double VirtualShapeGenerator::s_noteToDuration[128][8192] = { 0 };
bool VirtualShapeGenerator::s_init = false;
std::mutex VirtualShapeGenerator::s_mutex;

VirtualShapeGenerator::VirtualShapeGenerator() :
  _enableGlide(true), _enablePitch(true), _attack(0.5),
_sustain(1.0), _decay(1.05),
_release(1.05), _glideTime(5), _voices(2), _gain(.2),
_spread(1.01), _pitchAttack(12.), _pitchTime(.10), _pitchBend(0.0), _pitchRange(12.)
{
  VirtualShapeGenerator::InitStaticStuff();
}

void VirtualShapeGenerator::ProcessMidiMsg(const IMidiMsg& msg)
{
  switch (msg.StatusMsg())
  {
  case IMidiMsg::kNoteOn:
    startNote(msg);
    break;
  case IMidiMsg::kNoteOff:
    stopNote(msg);
    break;
  case IMidiMsg::kPolyAftertouch:
  case IMidiMsg::kControlChange:
  case IMidiMsg::kProgramChange:
  case IMidiMsg::kChannelAftertouch:
    return;
    break;
  case IMidiMsg::kPitchWheel:
    SetPitchBend(msg.PitchWheel()* _pitchRange);
    break;
  default:
    return;
  }
}

void VirtualShapeGenerator::ProcessBlock(sample** outputs, int nChannel,  int nFrames)
{
  double currentgain, time;
  for (int i = 0; i < nFrames; i++) {
    for (int c = 0; c < nChannel; c++) {
      outputs[c][i] = 0;
      _noteOnMutex.lock();
      for (auto it = _noteOn.begin(); it != _noteOn.end(); ++it) {
        time = _noteTab[*it]._time;
        if (time < _attack) currentgain = (1.0 - (_attack - time) / _attack);
        else if (time < _attack + _decay) currentgain = (_sustain + (_attack + _decay - time) / _decay * (1.0 - _sustain));
        else currentgain = _sustain;

        _noteTab[*it]._lastGain = currentgain;
        outputs[c][i] += _gain * currentgain * getShape(_noteTab[*it].normalizedPeriodLocation(c));
        _noteTab[*it].increment(_sampleDuration * _pitchBend, _glideTime, _pitchTime, _spread);
      }
      _noteOnMutex.unlock();
      _noteOffMutex.lock();
      for (auto it = _noteOff.begin(); it != _noteOff.end();) {
        if (_noteTab[*it]._releaseElapsed > _release) {
          it = _noteOff.erase(it);
        }
        else {
          currentgain = _gain * _noteTab[*it]._lastGain * (_release - _noteTab[*it]._releaseElapsed) / _release;
          outputs[c][i] += currentgain * getShape(_noteTab[*it].normalizedPeriodLocation(c));
          _noteTab[*it].increment(_sampleDuration * _pitchBend , _glideTime, _pitchTime, _spread);
          ++it;
        }
      }
      _noteOffMutex.unlock();
    }
  }
}

void VirtualShapeGenerator::Reset(const double& sampleRateValue)
{
  _sampleRate = sampleRateValue;
  _sampleDuration = 1.0 / sampleRateValue;
}

void VirtualShapeGenerator::EnableGlide(const bool& glideValue)
{
  _enableGlide = glideValue;
}

void VirtualShapeGenerator::SetAttack(const double& attackValue)
{
  _attack = attackValue;
}

void VirtualShapeGenerator::SetSustain(const double& sustainValue)
{
  _sustain = sustainValue;
}

void VirtualShapeGenerator::SetDecay(const double& decayValue)
{
  _decay = decayValue;
}

void VirtualShapeGenerator::SetRelease(const double& releaseValue)
{
  _release = releaseValue;
}

void VirtualShapeGenerator::SetGlideTime(const double& glideTimeValue)
{
  _glideTime = glideTimeValue;
}

void VirtualShapeGenerator::SetGain(const double& gainValue)
{
  _gain = gainValue;
}

void VirtualShapeGenerator::SetVoices(const int& voicesValue)
{
  _voices = voicesValue;
}

void VirtualShapeGenerator::SetSpread(const double& spreadValue)
{
  _spread = spreadValue;
}

void VirtualShapeGenerator::EnablePitch(const bool& enable)
{
  _enablePitch = enable;
}

void VirtualShapeGenerator::SetPitchAttack(const double& pitch)
{
  _pitchAttack = pitch;
}

void VirtualShapeGenerator::SetPitchTime(const double& pitchtime)
{
  _pitchTime = pitchtime;
}

void VirtualShapeGenerator::SetPitchBend(const double& pitchbend)
{
  _pitchBend = pow(2.0, pitchbend / 12.0);
}

void VirtualShapeGenerator::SetPitchRange(const double& pitchrange)
{
  _pitchRange = pitchrange;
}

void VirtualShapeGenerator::startNote(const IMidiMsg& msg)
{
  if (!_noteTab[msg.NoteNumber()]._isPlaying) {
    _noteOnMutex.lock();
    _noteOff.remove(msg.NoteNumber());
    _noteOnMutex.unlock();
    _noteTab[msg.NoteNumber()].reset(msg, _spread);
    if (_noteOn.size() >= _voices) {
      if (_enableGlide) {
        _noteTab[msg.NoteNumber()].start();
        _noteOnMutex.lock();
        _noteOn.push_back(msg.NoteNumber());
        _noteTab[msg.NoteNumber()].doGlideFrom(_noteTab[_noteOn.front()]);
        _noteTab[_noteOn.front()].forceStop();
        _noteOn.pop_front();
        _noteOnMutex.unlock();
      }
    }
    else {
      _noteTab[msg.NoteNumber()].start();
      _noteOnMutex.lock();
      _noteOn.push_back(msg.NoteNumber());
      if (_enablePitch)
        _noteTab[msg.NoteNumber()].doPitchAttack(_pitchAttack);
      _noteOnMutex.unlock();
    }
  }
}

void VirtualShapeGenerator::stopNote(const IMidiMsg& msg)
{
  if (_noteTab[msg.NoteNumber()]._isPlaying) {
    _noteOnMutex.lock();
    _noteOn.remove(msg.NoteNumber());
    _noteOnMutex.unlock();
    _noteOffMutex.lock();
    _noteOff.push_back(msg.NoteNumber());
    _noteOffMutex.unlock();
    _noteTab[msg.NoteNumber()].stop();
  }
}

double VirtualShapeGenerator::GetDurationFromMidi(const IMidiMsg& msg)
{
  return VirtualShapeGenerator::GetDurationFromNote(SimpleMidi(msg));
}

double VirtualShapeGenerator::GetDurationFromNote(const int& note, const int& pitch)
{
  return s_noteToDuration[note][pitch];
}

double VirtualShapeGenerator::GetDurationFromNote(const SimpleMidi& simple)
{
  return GetDurationFromNote(simple._note, simple._pitch);
}

void VirtualShapeGenerator::InitStaticStuff()
{
  std::lock_guard<std::mutex> guard(s_mutex);
  if (!s_init) {
    for (int i = 0; i < 128; i++) {
      for (int j = 0; j < 8192; j++) {
        s_noteToDuration[i][j] = 1.0 / (pow(2.0, (double(i) - 69.0  + double(j) / 8192.0) / 12.0) * 440.0);
      }
    }
  }
}

void VirtualShapeGenerator::SimpleMidi::fromDouble(const double& value)
{
  _note = int(value); _pitch = 8192.0 * (value- int(value));
  if (_note < 0) _note = 0;
  else if (_note > 127) _note = 127;

  if (_pitch < 0) _pitch = 0;
  else if (_pitch > 8191) _pitch = 8191;
}

double VirtualShapeGenerator::SimpleMidi::toDouble()
{
  return double(_note) + double(_pitch) / 8192.0;
}

void VirtualShapeGenerator::SimpleMidi::operator+=(const SimpleMidi& msg)
{
  _note += msg._note;
  _pitch += msg._pitch;
}

VirtualShapeGenerator::SimpleMidi::SimpleMidi(const IMidiMsg& msg)
{
  _note = msg.NoteNumber();
  if (msg.PitchWheel() < 0) {
    _note--;
    _pitch = (1.0 + msg.PitchWheel()) * 8192;
  }
  else {
    _pitch = msg.PitchWheel() * 8192;
  }
}

VirtualShapeGenerator::SimpleMidi::SimpleMidi(const SimpleMidi& msg) : _pitch(msg._pitch), _note(msg._note)
{
}

VirtualShapeGenerator::SimpleMidi::SimpleMidi() : _note(0), _pitch(0) {}

VirtualShapeGenerator::Note::Note() :
  _velocity(1.0), _time(0.0), _isPlaying(false), _isStopping(false),
_glide(false), _spread(1.0)
{
  _period[0] = GetDurationFromNote(_current) * _spread;
  _period[1] = GetDurationFromNote(_current) / _spread;
  _periodTime[0] = 0.0;
  _periodTime[1] = 0.0;
}

VirtualShapeGenerator::Note::Note(const IMidiMsg& msg, const double& spreadValue) :
  _velocity(msg.Velocity()), _time(0.0), _isPlaying(false), _isStopping(false),
_glide(false), _target(msg), _spread(spreadValue)
{
  _current = _target;
  _period[0] = GetDurationFromNote(_current) * _spread;
  _period[1] = GetDurationFromNote(_current) / _spread;
  _periodTime[0] = 0.0;
  _periodTime[1] = 0.0;
}

void VirtualShapeGenerator::Note::reset(const IMidiMsg& msg, const double& spreadValue)
{
  _spread = spreadValue;
  *this = Note(msg, spreadValue);
  _time = 0.0;
  _period[0] = GetDurationFromNote(_current) * _spread;
  _period[1] = GetDurationFromNote(_current) / _spread;
  _periodTime[0] = 0.0;
  _periodTime[1] = 0.0;
  forceStop();
}

void VirtualShapeGenerator::Note::start()
{
  _time = 0.0;
  _periodTime[0] = 0.0;
  _periodTime[1] = 0.0;
  _isPlaying = true;
  _isStopping = false;
}

void VirtualShapeGenerator::Note::stop()
{
  _releaseElapsed = 0.0;
  _isPlaying = false;
  _isStopping = true;
}

double VirtualShapeGenerator::Note::normalizedPeriodLocation(const int& channel) const
{
  return _periodTime[channel] / _period[channel];
}

void VirtualShapeGenerator::Note::forceStop()
{
  _isPlaying = false;
  _isStopping = false;
}

void VirtualShapeGenerator::Note::doGlideFrom(const Note& note)
{
  _glideElapsed = 0.0;
  _time = note._time;
  _glide = true;
  _current = note._current;
  _endNote = _target.toDouble();
  _startNote = _current.toDouble();
  _period[0] = GetDurationFromNote(_current) * _spread;
  _period[1] = GetDurationFromNote(_current) / _spread;
  _isPlaying = true;
  _isStopping = false;
}

void VirtualShapeGenerator::Note::doPitchAttack(const double& pitch)
{
  _pitchElapsed = 0.0;
  _time = 0.0;
  _pitch = true;
  _endNote = _target.toDouble();
  _startNote = _current.toDouble() + pitch;
  _current.fromDouble(_startNote);
  _period[0] = GetDurationFromNote(_current) * _spread;
  _period[1] = GetDurationFromNote(_current) / _spread;
  _isPlaying = true;
  _isStopping = false;
}

void VirtualShapeGenerator::Note::increment(const double& timestep, const double & glideTime, const double& pitchTime, const double& spreadValue)
{
  if (_glide) {
    _glide = legato(_glideElapsed, glideTime);
      _glideElapsed += timestep;
  }
  else if (_pitch) {
    _pitch = legato(_pitchElapsed, pitchTime);
    _pitchElapsed += timestep;
  }
  _time += timestep;
  if (_isStopping)
    _releaseElapsed += timestep;
  for (int i = 0; i < 2; i++) {
    _periodTime[i] += timestep;

    while (_periodTime[i] > _period[i])
      _periodTime[i] -= _period[i];
  }
}

bool VirtualShapeGenerator::Note::legato(const double& elapsed, const double& duration)
{
  if (elapsed > duration) {
    _current = _target;
    return false;
  }
  else {
    _current.fromDouble(_startNote + (_endNote - _startNote) * elapsed / duration);
    _period[0] = GetDurationFromNote(_current) * _spread;
    _period[1] = GetDurationFromNote(_current) / _spread;
    return true;
  }
}

