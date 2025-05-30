import os
from pydub import AudioSegment


input_folder = 'music_synthesis/audio/piano0.5'
output_folder = 'music_synthesis/audio/piano0.3'
start_time = 0.0
end_time = 0.3

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".wav"):
        filepath = os.path.join(input_folder, filename)
        audio = AudioSegment.from_wav(filepath)

        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        end_ms = min(end_ms, len(audio))

        trimmed_audio = audio[start_ms:end_ms]
        output_path = os.path.join(output_folder, f"{filename}")
        trimmed_audio.export(output_path, format="wav")

        print(f"Файл {filename} обрізано та збережено як {output_path}")
