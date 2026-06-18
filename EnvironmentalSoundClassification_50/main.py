import io
import torch
import uvicorn
import numpy as np
import torch.nn as nn
import soundfile as sf
import torch.nn.functional as F
import streamlit as st
from torchaudio import transforms
from fastapi import FastAPI, HTTPException, UploadFile, File


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 50)
        )

    def forward(self, audio):
        audio = audio.unsqueeze(1)
        audio = self.first(audio)
        audio = self.second(audio)
        return audio

sample_rate = 22050
n_mels = 32
max_len = 500

transform = nn.Sequential(
    transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels),
    transforms.AmplitudeToDB()
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleCNN().to(device)
model.load_state_dict(torch.load(
    'model_EnvironmentalSoundClassification_50.pth',
    map_location=device
))
model.eval()

classes = torch.load(
    'labels_EnvironmentalSoundClassification_50.pth',
    map_location='cpu',
    weights_only=False
)

if isinstance(classes, torch.Tensor):
    classes = classes.tolist()
elif isinstance(classes, dict):
    classes = list(classes.values())

print(f"Модель загружена. Классов: {len(classes)}")


def change_audio(waveform, sr):
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        resample = transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resample(waveform)

    spec = transform(waveform)

    if spec.dim() == 3 and spec.shape[0] == 1:
        spec = spec.squeeze(0)

    if spec.shape[-1] > max_len:
        spec = spec[..., :max_len]
    elif spec.shape[-1] < max_len:
        spec = F.pad(spec, (0, max_len - spec.shape[-1]))

    return spec.unsqueeze(0).to(device)


audio_app = FastAPI(title='Environment sounds')

@audio_app.post('/predict/')
async def predict_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='File not found')

        wf, sr = sf.read(io.BytesIO(data), dtype='float32')

        spec = change_audio(wf, sr)

        with torch.no_grad():
            y_pred = model(spec)
            pred_ind = torch.argmax(y_pred, dim=1).item()
            pred_class = classes[pred_ind]
            confidence = torch.softmax(y_pred, dim=1)[0][pred_ind].item()

        return {
            "index": pred_ind,
            "sound_type": pred_class,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(audio_app, host='127.0.0.1', port=8000)


# st.title('Environment Sounds Classifier')
# st.text('Upload audio (.wav) to recognize sound')
#
# file = st.file_uploader('Upload a file', type=['wav'])
#
# if not file:
#     st.warning('Upload a file')
# else:
#     st.audio(file)
#     if st.button('Recognize'):
#         try:
#             data = file.read()
#
#             wf, sr = sf.read(io.BytesIO(data), dtype='float32')
#
#             spec = change_audio(wf, sr)
#
#             with torch.no_grad():
#                 y_pred = model(spec)
#                 pred_ind = torch.argmax(y_pred, dim=1).item()
#                 pred_class = classes[pred_ind]
#                 confidence = torch.softmax(y_pred, dim=1)[0][pred_ind].item()
#
#             st.success(f'Index: {pred_ind}, Sound type: {pred_class}')
#             st.info(f'Confidence: {confidence:.1%}')
#
#         except Exception as e:
#             st.warning(f'Error: {e}')
