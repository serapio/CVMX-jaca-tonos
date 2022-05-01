from io import BytesIO
from typing import Tuple
import wave
import gradio as gr
import numpy as np
from pydub.audio_segment import AudioSegment
import requests
from os.path import exists
from stt import Model


# download model
storage_url = "https://coqui.gateway.scarf.sh/mixtec/jemeyer/v1.0.0"
model_name = "model.tflite"
model_link = f"{storage_url}/{model_name}"


def client(audio_data: np.array, sample_rate: int, use_scorer=False):
    output_audio = _convert_audio(audio_data, sample_rate)

    fin = wave.open(output_audio, 'rb')
    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    fin.close()

    ds = Model(model_name)
    if use_scorer:
        ds.enableExternalScorer("kenlm.scorer")

    result = ds.stt(audio)

    return result


def download(url, file_name):
    if not exists(file_name):
        print(f"Downloading {file_name}")
        r = requests.get(url, allow_redirects=True)
        with open(file_name, 'wb') as file:
            file.write(r.content)
    else:
        print(f"Found {file_name}. Skipping download...")


def stt(audio: Tuple[int, np.array]):
    sample_rate, audio = audio
    use_scorer = False

    if sample_rate != 16000:
        raise ValueError("Incorrect sample rate.")

    recognized_result = client(audio, sample_rate, use_scorer)

    return recognized_result


def _convert_audio(audio_data: np.array, sample_rate: int):
    source_audio = BytesIO()
    source_audio.write(audio_data)
    source_audio.seek(0)
    output_audio = BytesIO()
    wav_file = AudioSegment.from_raw(
        source_audio,
        channels=1,
        sample_width=2,
        frame_rate=sample_rate
    )
    wav_file.set_frame_rate(16000).set_channels(
        1).export(output_audio, "wav", codec="pcm_s16le")
    output_audio.seek(0)
    return output_audio


iface = gr.Interface(
    fn=stt,
    inputs=[
        gr.inputs.Audio(type="numpy",
                        label=None, optional=False),
    ],
    outputs=gr.outputs.Textbox(label="Output"),
    title="Coqui STT Yoloxochitl Mixtec",
    theme="huggingface",
    description="Speech-to-text demo for Yoloxochitl Mixtec, using the model trained by Josh Meyer on [the corpus compiled by Rey Castillo and collaborators](https://www.openslr.org/89). This demo is based on the [Ukrainian STT demo](https://huggingface.co/spaces/robinhad/ukrainian-stt).",
)

download(model_link, model_name)
iface.launch()
