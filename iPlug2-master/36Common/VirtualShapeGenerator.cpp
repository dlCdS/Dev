#include "VirtualShapeGenerator.h"

double VirtualShapeGenerator::s_noteToDuration[128][8192] = { 0 };
bool VirtualShapeGenerator::s_init = false;
std::mutex VirtualShapeGenerator::s_mutex;

VirtualShapeGenerator::VirtualShapeGenerator() : enableGlide(true), attack(0.5), sustain(1.0), decay(1.05), release(1.05), glideTime(5), voices(2), gain(.2)
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
      noteOnMutex.lock();
      for (auto it = noteOn.begin(); it != noteOn.end(); ++it) {
        time = noteTab[*it].time;
        if (time < attack) currentgain = (1.0 - (attack - time) / attack);
        else if (time < attack + decay) currentgain = (sustain + (attack + decay - time) / decay * (1.0 - sustain));
        else currentgain = sustain;

        noteTab[*it].lastGain = currentgain;
        outputs[c][i] += gain * currentgain * getShape(noteTab[*it].normalizedPeriodLocation());
        noteTab[*it].increment(sampleDuration, glideTime);
      }
      noteOnMutex.unlock();
      noteOffMutex.lock();
      for (auto it = noteOff.begin(); it != noteOff.end();) {
        if (noteTab[*it].releaseTime > release) {
          it = noteOff.erase(it);
        }
        else {
          currentgain = gain * noteTab[*it].lastGain * (release - noteTab[*it].releaseTime) / release;
          outputs[c][i] += currentgain * getShape(noteTab[*it].normalizedPeriodLocation());
          noteTab[*it].increment(sampleDuration, glideTime);
          ++it;
        }
      }
      noteOffMutex.unlock();
    }
  }
}

void VirtualShapeGenerator::Reset(const double& sampleRateValue)
{
  sampleRate = sampleRateValue;
  sampleDuration = 1.0 / sampleRateValue;
}

void VirtualShapeGenerator::SetGlide(const bool& glideValue)
{
  enableGlide = glideValue;
}

void VirtualShapeGenerator::SetAttack(const double& attackValue)
{
  attack = attackValue;
}

void VirtualShapeGenerator::SetSustain(const double& sustainValue)
{
  sustain = sustainValue;
}

void VirtualShapeGenerator::SetDecay(const double& decayValue)
{
  decay = decayValue;
}

void VirtualShapeGenerator::SetRelease(const double& releaseValue)
{
  release = releaseValue;
}

void VirtualShapeGenerator::SetGlideTime(const double& glideTimeValue)
{
  glideTime = glideTimeValue;
}

void VirtualShapeGenerator::SetGain(const double& gainValue)
{
  gain = gainValue;
}

void VirtualShapeGenerator::SetVoices(const int& voicesValue)
{
  voices = voicesValue;
}

void VirtualShapeGenerator::startNote(const IMidiMsg& msg)
{
  if (!noteTab[msg.NoteNumber()].isPlaying) {
    noteOnMutex.lock();
    noteOff.remove(msg.NoteNumber());
    noteOnMutex.unlock();
    noteTab[msg.NoteNumber()].reset(msg);
    if (noteOn.size() >= voices) {
      if (enableGlide) {
        noteTab[msg.NoteNumber()].start();
        noteOnMutex.lock();
        noteOn.push_back(msg.NoteNumber());
        noteTab[msg.NoteNumber()].doGlideFrom(noteTab[noteOn.front()]);
        noteTab[noteOn.front()].forceStop();
        noteOn.pop_front();
        noteOnMutex.unlock();
      }
    }
    else {
      noteTab[msg.NoteNumber()].start();
      noteOnMutex.lock();
      noteOn.push_back(msg.NoteNumber());
      noteOnMutex.unlock();
    }
  }
}

void VirtualShapeGenerator::stopNote(const IMidiMsg& msg)
{
  if (noteTab[msg.NoteNumber()].isPlaying) {
    noteOnMutex.lock();
    noteOn.remove(msg.NoteNumber());
    noteOnMutex.unlock();
    noteOffMutex.lock();
    noteOff.push_back(msg.NoteNumber());
    noteOffMutex.unlock();
    noteTab[msg.NoteNumber()].stop();
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
  return GetDurationFromNote(simple.note, simple.pitch);
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
  note = int(value); pitch = 8192.0 * (value- int(value));
  if (note < 0) note = 0;
  else if (note > 127) note = 127;

  if (pitch < 0) pitch = 0;
  else if (pitch > 8191) pitch = 8191;
}

double VirtualShapeGenerator::SimpleMidi::toDouble()
{
  return double(note) + double(pitch) / 8192.0;
}

void VirtualShapeGenerator::SimpleMidi::operator+=(const SimpleMidi& msg)
{
  note += msg.note;
  pitch += msg.pitch;
}

VirtualShapeGenerator::SimpleMidi::SimpleMidi(const IMidiMsg& msg)
{
  note = msg.NoteNumber();
  if (msg.PitchWheel() < 0) {
    note--;
    pitch = (1.0 + msg.PitchWheel()) * 8192;
  }
  else {
    pitch = msg.PitchWheel() * 8192;
  }
}

VirtualShapeGenerator::SimpleMidi::SimpleMidi(const SimpleMidi& msg) : pitch(msg.pitch), note(msg.note)
{
}

VirtualShapeGenerator::SimpleMidi::SimpleMidi() : note(0), pitch(0) {}

VirtualShapeGenerator::Note::Note() : velocity(1.0), time(0.0), isPlaying(false), isStopping(false),
periodTime(0.0), glide(false), period(GetDurationFromNote(current)) {}

VirtualShapeGenerator::Note::Note(const IMidiMsg& msg) : velocity(msg.Velocity()), time(0.0), isPlaying(false), isStopping(false),
periodTime(0.0), glide(false), target(msg) {
  current = target;
  period = GetDurationFromNote(current);
}

void VirtualShapeGenerator::Note::reset(const IMidiMsg& msg)
{
  *this = Note(msg);
  time = 0.0;
  periodTime = 0.0;
  period = GetDurationFromNote(current);
  forceStop();
}

void VirtualShapeGenerator::Note::start()
{
  time = 0.0;
  periodTime = 0.0;
  isPlaying = true;
  isStopping = false;
}

void VirtualShapeGenerator::Note::stop()
{
  releaseTime = 0.0;
  isPlaying = false;
  isStopping = true;
}

double VirtualShapeGenerator::Note::normalizedPeriodLocation() const
{
  return periodTime / period;
}

void VirtualShapeGenerator::Note::forceStop()
{
  isPlaying = false;
  isStopping = false;
}


void VirtualShapeGenerator::Note::doGlideFrom(const IMidiMsg& msg) {
  doGlideFrom(Note(msg));
}

void VirtualShapeGenerator::Note::doGlideFrom(const Note& note)
{
  glidetime = 0.0;
  time = note.time;
  glide = true;
  current = note.current;
  endNote = target.toDouble();
  startNote = current.toDouble();
  period = GetDurationFromNote(current);
  isPlaying = true;
  isStopping = false;
}

void VirtualShapeGenerator::Note::increment(const double& timestep, const double & glideTime)
{
  if (glide) {
    if (glidetime > glideTime) {
      glide = false;
      current = target;
    }
    else {
      current.fromDouble(startNote + (endNote - startNote) * glidetime / glideTime);
      period = GetDurationFromNote(current);
      glidetime += timestep;
    }
  }
  time += timestep;
  periodTime += timestep;
  if (isStopping)
    releaseTime += timestep;
  while (periodTime > period)
    periodTime -= period;
}

