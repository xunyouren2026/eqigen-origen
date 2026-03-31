import subprocess
import numpy as np
import soundfile as sf


def generate_tts(text, output_path, voice="zh-CN-XiaoxiaoNeural"):
    try:
        subprocess.run(['edge-tts', '--text', text, '--voice',
                       voice, '--write-media', output_path], check=True)
    except:
        import wave
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b'\x00' * 24000 * 2)
    return output_path


def add_background_music(video_path, music_path, output_path):
    cmd = ['ffmpeg', '-i', video_path, '-i', music_path,
           '-filter_complex', 'amix', '-y', output_path]
    subprocess.run(cmd, check=True)
    return output_path
