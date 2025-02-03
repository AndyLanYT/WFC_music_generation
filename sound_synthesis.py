import math
import wave


FRAMES_PER_SECOND = 22050
frequency = 493.8833
duration = 2

def beat(frequency, num_seconds):
    for frame in range(round(num_seconds * FRAMES_PER_SECOND)):
        time = frame / FRAMES_PER_SECOND
        amplitude = math.sin(2 * math.pi * frequency * time)
        
        yield round((amplitude + 1) / 2 * 255)


with wave.open(f"music_synthesis/audio/B4.wav", mode="wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(1)
    wav_file.setframerate(FRAMES_PER_SECOND)
    wav_file.writeframes(bytes(beat(frequency, duration)))
