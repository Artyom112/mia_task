"""
Сервис верификации диктора на базе ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb).

Принимает два аудиофайла, извлекает голосовые эмбеддинги и по косинусному сходству
между ними определяет, принадлежат ли записи одному и тому же диктору. Интерфейс
собран на Gradio, веб-страница с двумя полями загрузки аудио и результатом сравнения.

Запуск: `python app.py`
"""

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)


def load_and_preprocess(path, target_sr=16000):
    """
    Загружает аудиофайл с диска и приводит его к формату, который ожидает модель:
    один (моно) канал и частота дискретизации 16 кГц.

    Параметры:
        path (str): путь к аудиофайлу (.wav, .mp3 и т.п.)
        target_sr (int): целевая частота дискретизации в Гц (модель обучена на 16000)

    Возвращает:
        torch.Tensor формы (1, num_samples) - моно-сигнал на частоте target_sr.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav


def get_embedding(path):
    """
    Строит голосовой эмбеддинг для аудиофайла.

    Параметры:
        path (str): путь к аудиофайлу.

    Возвращает:
        torch.Tensor: одномерный вектор эмбеддинга (без лишних размерностей).
    """
    wav = load_and_preprocess(path)
    emb = classifier.encode_batch(wav)
    return emb.squeeze()


def verify(path1, path2, threshold=0.3088, scale=11.41):
    """
    Сравнивает два аудиофайла и определяет, принадлежат ли они одному говорящему.

    Параметры:
        path1 (str): путь к первому аудиофайлу.
        path2 (str): путь ко второму аудиофайлу.
        threshold (float): порог косинусного сходства, выше которого голоса
            считаются совпадающими (подобран эмпирически для этой модели).
        scale (float): коэффициент "крутизны" сигмоиды при переводе score
            в доверительную вероятность (confidence).

    Возвращает:
        tuple: (label, confidence, score)
            label (str): "SAME" (один голос) или "DIFFERENT" (разные голоса)
            confidence (float): условная "уверенность" модели в диапазоне 0..1
            score (float): исходное косинусное сходство эмбеддингов (-1..1)
    """
    e1, e2 = get_embedding(path1), get_embedding(path2)

    score = torch.nn.functional.cosine_similarity(e1, e2, dim=0).item()
    confidence = torch.sigmoid(torch.tensor((score - threshold) * scale)).item()
    label = "SAME" if score > threshold else "DIFFERENT"

    return label, confidence, score


def predict(audio1, audio2):
    """
    Функция-обработчик для Gradio: принимает пути к двум аудиофайлам,
    вызывает verify() и форматирует результат для отображения в интерфейсе.

    Параметры:
        audio1 (str): путь к первому аудиофайлу (передаёт Gradio).
        audio2 (str): путь ко второму аудиофайлу (передаёт Gradio).

    Возвращает:
        tuple[str, str, str]: (результат на русском, уверенность, сырой score) -
        все строки, т.к. выводятся в текстовые поля интерфейса.
    """
    label, conf, raw_score = verify(audio1, audio2)

    label_ru = "ОДИН И ТОТ ЖЕ ГОЛОС" if label == "SAME" else "РАЗНЫЕ ГОЛОСА"

    return label_ru, f"{conf:.3f}", f"{raw_score:.3f}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Audio(type="filepath", label="Аудио 1"),
        gr.Audio(type="filepath", label="Аудио 2"),
    ],
    outputs=[
        gr.Textbox(label="Результат"),
        gr.Textbox(label="Уверенность"),
        gr.Textbox(label="Косинусное сходство"),
    ],
    title="Верификация диктора",
    description="Загрузите два аудиофайла, чтобы проверить, принадлежат ли они одному и тому же голосу.",
)

if __name__ == "__main__":
    demo.launch()
