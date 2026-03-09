import torch
import scipy
from diffusers import AudioLDM2Pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2-music",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

audio = pipe(
    "pop music",
    num_inference_steps=20,
    audio_length_in_s=3.5
).audios[0]

scipy.io.wavfile.write("test.wav", rate=16000, data=audio)

print("done")