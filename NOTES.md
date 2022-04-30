

# Things that might be relevant

## Trained models

ESPnet model for Yoloxochitl Mixtec 
 - Huggingface Hub page
 - Model source code https://github.com/espnet/espnet/tree/master/egs/yoloxochitl_mixtec/asr1
 - Colab notebook to setup and apply the model https://colab.research.google.com/drive/1ieoW2b3ERydjaaWuhVPBP_v2QqqWsC1Q?usp=sharing
 
Coqui model for Yoloxochitl Mixtec
 - Huggingface Hub page
 - Coqui page
 - Colab notebook to setup and apply the model https://colab.research.google.com/drive/1b1SujEGC_F3XhvUCuUyZK_tyUkEaFZ7D?usp=sharing#scrollTo=6IvRFke4Ckpz

Spanish ASR models
 - XLS-R model based on CV8 with LM https://huggingface.co/jonatasgrosman/wav2vec2-xls-r-1b-spanish
 - XLSR model based on CV6 with LM https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish
 - XLSR model based on Librispeech https://huggingface.co/IIC/wav2vec2-spanish-multilibrispeech

Speechbrain Language identification on Common Language (from Common Voice 6/7?)
 - source code https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonLanguage
 - HF Hub model page https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa
 - HF Hub space https://huggingface.co/spaces/akhaliq/Speechbrain-audio-classification

Speechbrain Language identification on VoxLingua
 - source code https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxLingua107/lang_id
 - HF Hub model page https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa


## Corpora

OpenSLR89 https://www.openslr.org/89/

Common Language https://huggingface.co/datasets/common_language

VoxLingua http://bark.phon.ioc.ee/voxlingua107/

Multilibrispeech https://huggingface.co/datasets/multilingual_librispeech


# Possible demos

## Simple categorization of utterances

A few example files are provided for each language, and the user can record their own.
The predicted confidence of each class label is shown.

## Segmentation and identification

Recordings with alternating languages in a single audio file, provided examples or the user can record.
Some voice activity detection to split the audio, then predict language of each piece

## Identication and transcription

Example files for each language separately.
The lang-id model predicts what language it is.
The corresponding ASR model produces a transcript.

## Segmentation, identification and transcription

Recordings with alternating languages in a single audio file.
Use voice activity detection to split the audio, then predict the language of each piece
Use the corresponding ASR model to produce a transcript of each piece to display.