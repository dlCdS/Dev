#include "VirtualShapeGenerator.h"

double VirtualShapeGenerator::s_noteToFreq[128][8192] = { 0 };
bool VirtualShapeGenerator::s_init = false;
std::mutex VirtualShapeGenerator::s_mutex;

VirtualShapeGenerator::VirtualShapeGenerator()
{
  VirtualShapeGenerator::InitStaticStuff();
}

void VirtualShapeGenerator::ProcessMidiMsg(const IMidiMsg& msg)
{

}

double VirtualShapeGenerator::GetFreqFromMidi(const IMidiMsg& msg)
{
  VirtualShapeGenerator::GetFreqFromNote(SimpleMidi(msg));
}

double VirtualShapeGenerator::GetFreqFromNote(const int& note, const int& pitch)
{
  return s_noteToFreq[note][pitch];
}

double VirtualShapeGenerator::GetFreqFromNote(const SimpleMidi& simple)
{
  return GetFreqFromNote(simple.note, simple.pitch);
}

void VirtualShapeGenerator::InitStaticStuff()
{
  std::lock_guard<std::mutex> guard(s_mutex);
  if (!s_init) {
    for (int i = 0; i < 128; i++) {
      for (int j = 0; j < 8192; j++) {
        s_noteToFreq[i][j] = pow(2.0, (double(i) - 69.0  + double(j) / 8192.0) / 12.0) * 440.0;
      }
    }
  }
}

void VirtualShapeGenerator::SimpleMidi::fromDouble(const double& value)
{
  note = int(value); pitch = 8192.0 * value;
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

VirtualShapeGenerator::SimpleMidi::SimpleMidi() : note(0), pitch(0) {}

VirtualShapeGenerator::Note::Note() : velocity(1.0), time(0.0), glide(false), freq(GetFreqFromNote(current)) {}

VirtualShapeGenerator::Note::Note(const IMidiMsg& msg) : velocity(1.0), time(0.0), glide(false), target(msg), current(target), freq(GetFreqFromNote(current)) {}


void VirtualShapeGenerator::Note::doGlide(const IMidiMsg& msg) {
  time = 0.0; glide = true; current = SimpleMidi(msg); glidepos = current.toDouble(); freq = GetFreqFromNote(current);
}

