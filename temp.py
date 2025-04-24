import os
import torch
import librosa
import numpy as np
import scipy.signal as sps
import wave
import pyaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoProcessor, AutoModelForCTC
from jiwer import wer, cer

# ================== AUDIO PROCESSING ==================

def detect_noise_level(audio, sr):
    rms = np.sqrt(np.mean(audio ** 2))
    noise_threshold = np.percentile(np.abs(audio), 10)
    snr = 20 * np.log10(rms / (noise_threshold + 1e-6))
    print(f"[INFO] SNR: {snr:.2f} dB")
    return snr

def apply_filter(audio, sr, filter_type='low', cutoff_freq=3000, order=6):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff_freq / nyquist
    b, a = sps.butter(order, norm_cutoff, btype=filter_type, analog=False)
    return sps.filtfilt(b, a, audio)

def load_audio(file_path, sr=16000):
    audio, rate = librosa.load(file_path, sr=sr)
    return audio, sr

# ================== TRANSCRIPTION ==================

def transcribe_wav2vec(audio, sr):
    print("[INFO] Transkrypcja: Wav2Vec2")
    model_name = "facebook/wav2vec2-large-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()

def transcribe_mms(audio, sr):
    print("[INFO] Transkrypcja: facebook/mms-1b-all")
    model_name = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()

# ================== RECORDING ==================

def record_audio(filename="nagranie.wav", duration=5, sr=16000):
    chunk = 1024
    fmt = pyaudio.paInt16
    channels = 1

    pa = pyaudio.PyAudio()
    stream = pa.open(format=fmt, channels=channels, rate=sr, input=True, frames_per_buffer=chunk)

    print("🎙️ Nagrywanie...")
    frames = []
    for _ in range(int(sr / chunk * duration)):
        frames.append(stream.read(chunk))
    print("✔️ Nagrywanie zakończone.")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pa.get_sample_size(fmt))
    wf.setframerate(sr)
    wf.writeframes(b''.join(frames))
    wf.close()

# ================== COMPARISON ==================

def compare_transcriptions(ref, hyp):
    wer_score = round(wer(ref, hyp) * 100, 2)
    cer_score = round(cer(ref, hyp) * 100, 2)
    return wer_score, cer_score

# ================== MAIN EXECUTION ==================

def main():
    print("==== SYSTEM TRANSKRYPCJI AUDIO ====\n")
    mode = input("Wybierz tryb [plik/mikrofon]: ").strip().lower()

    if mode == "plik":
        file_path = input("Podaj ścieżkę do pliku audio (WAV/MP3): ").strip()
    elif mode == "mikrofon":
        duration = int(input("Czas nagrania w sekundach: "))
        file_path = "nagranie.wav"
        record_audio(file_path, duration)
    else:
        print("Nieznany tryb.")
        return

    # Wczytanie i obróbka audio
    audio, sr = load_audio(file_path)
    detect_noise_level(audio, sr)

    use_filter = input("Zastosować filtrację? [t/n]: ").strip().lower()
    if use_filter == 't':
        typ = input("Typ filtru [low/high]: ").strip()
        freq = int(input("Częstotliwość odcięcia (Hz): "))
        audio = apply_filter(audio, sr, typ, freq)

    # Transkrypcje
    transcription_wav2vec = transcribe_wav2vec(audio, sr)
    transcription_mms = transcribe_mms(audio, sr)

    print("\n--- Wav2Vec2 ---")
    print(transcription_wav2vec)

    print("\n--- MMS ---")
    print(transcription_mms)

    manual = input("\nWklej własny transkrypt do porównania: ").strip().lower()

    wer_wav2vec, cer_wav2vec = compare_transcriptions(manual, transcription_wav2vec)
    wer_mms, cer_mms = compare_transcriptions(manual, transcription_mms)

    with open("raport.txt", "w", encoding="utf-8") as f:
        f.write("=== Raport Transkrypcji ===\n\n")
        f.write("--- Wav2Vec2 ---\n")
        f.write(transcription_wav2vec + "\n")
        f.write(f"WER: {wer_wav2vec} %\nCER: {cer_wav2vec} %\n\n")

        f.write("--- MMS ---\n")
        f.write(transcription_mms + "\n")
        f.write(f"WER: {wer_mms} %\nCER: {cer_mms} %\n\n")

        f.write("=== Transkrypt referencyjny ===\n")
        f.write(manual + "\n")

    print("\n✅ Wyniki zapisane do pliku: raport.txt")

if __name__ == "__main__":
    main()
