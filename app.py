from io import BytesIO
from typing import Tuple
import wave
import gradio as gr
import numpy as np
from pydub.audio_segment import AudioSegment
import requests
from os.path import exists
from stt import Model

import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor

import torchaudio
from speechbrain.pretrained import EncoderClassifier

# initialize language ID model
lang_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-commonlanguage_ecapa", 
    savedir="pretrained_models/lang-id-commonlanguage_ecapa"
)

def load_hf_model(model_path="facebook/wav2vec2-large-robust-ft-swbd-300h"):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path)
    return processor, model

# download STT model
model_info = {
    "mixteco": ("https://coqui.gateway.scarf.sh/mixtec/jemeyer/v1.0.0/model.tflite", "mixtec.tflite"),
    "chatino": ("https://coqui.gateway.scarf.sh/chatino/bozden/v1.0.0/model.tflite", "chatino.tflite"),
    "totonaco": ("https://coqui.gateway.scarf.sh/totonac/bozden/v1.0.0/model.tflite", "totonac.tflite"),
    "español": ("jonatasgrosman/wav2vec2-large-xlsr-53-spanish", "spanish_xlsr"),
    "inglés": ("facebook/wav2vec2-large-robust-ft-swbd-300h", "english_xlsr"),
}

STT_MODELS = {lang: load_hf_model(model_info[lang][0]) for lang in ("español",)}


def client(audio_data: np.array, sample_rate: int, default_lang: str):
    output_audio = _convert_audio(audio_data, sample_rate)
    waveform, _ = torchaudio.load(output_audio)
    out_prob, score, index, text_lab = lang_classifier.classify_batch(waveform)
    text_lab = text_lab[0]

    output_audio.seek(0)
    fin = wave.open(output_audio, 'rb')
    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    fin.close()
    print(default_lang, text_lab)

    if text_lab == 'Spanish':
        text_lab = 'español'
        processor, model = STT_MODELS['español']
        inputs = processor(waveform)
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        result = processor.decode(torch.argmax(logits, dim=-1).cpu().tolist())

    else:
        text_lab = default_lang
        ds = STT_MODELS[default_lang]
        result = ds.stt(audio)

    return f"{text_lab}: {result}"


def load_coqui_models(language):

    model_path, file_name = model_info.get(language, ("", ""))

    if not exists(file_name):
        print(f"Downloading {model_path}")
        r = requests.get(model_path, allow_redirects=True)
        with open(file_name, 'wb') as file:
            file.write(r.content)
    else:
        print(f"Found {file_name}. Skipping download...")
    return Model(file_name)

for lang in ('mixteco', 'chatino', 'totonaco'):
    STT_MODELS[lang] = load_coqui_models(lang)



def stt(default_lang: str, audio: Tuple[int, np.array]):
    sample_rate, audio = audio
    use_scorer = False

    recognized_result = client(audio, sample_rate, default_lang)

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
    wav_file.set_frame_rate(16000).set_channels(1).export(output_audio, "wav", codec="pcm_s16le")
    output_audio.seek(0)
    return output_audio


iface = gr.Interface(
    fn=stt,
    inputs=[
        gr.inputs.Radio(choices=("chatino", "mixteco", "totonaco"), default="mixteco", label="Lengua principal"),
        gr.inputs.Audio(type="numpy", label="Audio", optional=False),
    ],
    outputs=gr.outputs.Textbox(label="Output"),
    title="Coqui STT de Chatino, Mixteco, y Totonaco",
    theme="huggingface",
    description="Prueba de identificar frases de español en grabaciones de una lengua indígena, y prover el texto de cada una"
    examples=[["mixteco", "espanol1-Yolox_BotFl_CTB501-FEF537-EGS503_40202-Acanthaceae-Ruellia_2017-01-05-h-espanol.wav"],
            ["mixteco", "espanol2-Yolox_BotFl_CTB501-FEF537-EGS503_40202-Acanthaceae-Ruellia_2017-01-05-h.wav"],
            ["mixteco", "mixteco1-Yolox_BotFl_CTB501-FEF537-EGS503_40202-Acanthaceae-Ruellia_2017-01-05-h.wav"],
            ["mixteco", "mixteco2-Yolox_BotFl_CTB501-FEF537-EGS503_40202-Acanthaceae-Ruellia_2017-01-05-h.wav"],
            ["totonaco", "totonaco1-Zongo_Botan_Acanthaceae-Justicia-spicigera_SLC388-IPN389_2018-07-26-i.wav"],
            ["totonaco", "totonaco2-Zongo_Botan_Acanthaceae-Justicia-spicigera_SLC388-IPN389_2018-07-26-i.wav"]]
    article="Chatino: Prueba de dictado a texto para el chatino de la sierra (Quiahije) "
                " usando [el modelo entrenado por Bülent Özden](https://coqui.ai/chatino/bozden/v1.0.0)"
                " con [los datos recopilados por Hilaria Cruz y sys colaboradores](https://gorilla.linguistlist.org/code/ctp/)"
                "\n\n"
                "Mixteco: Prueba de dictado a texto para el mixteco de Yoloxochitl,"
                " usando [el modelo entrenado por Josh Meyer](https://coqui.ai/mixtec/jemeyer/v1.0.0/)"
                " con [los datos recopilados por Rey Castillo, Jonathan Amith y sus colaboradores](https://www.openslr.org/89)."
                " Esta prueba es basada en la de [Ukraniano](https://huggingface.co/spaces/robinhad/ukrainian-stt)."
                " \n\n"
                "Totonaco: Prueba de dictado a texto para el totonaco de la sierra,"
                " usando [el modelo entrenado por Bülent Özden](https://coqui.ai/totonac/bozden/v1.0.0)"
                " con [los datos recopilados por Osbel López Francisco y Jonathan Amith](https://www.openslr.org/107)."
                " \n\n"
                "Los ejemplos vienen del proyecto [DEMCA](https://demca.mesolex.org/). "
                " Esta prueba es basada en la de [Ukraniano](https://huggingface.co/spaces/robinhad/ukrainian-stt)."
)


iface.launch()
