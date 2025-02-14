import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import threading
import queue
import uuid
from pynput.keyboard import Controller, Key
import time

# Configurações de áudio
SAMPLE_RATE = 16000
BLOCK_DURATION = 0.5
MAX_WORKERS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Carregando o modelo Whisper
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="portuguese")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)

# Fila para armazenar blocos de áudio
audio_queue = queue.Queue()

# Variável compartilhada entre threads
current_command = ""
lock = threading.Lock()

# Controlador de teclado
keyboard = Controller()

def transcribe_audio_block(audio_block, audio_id):
    try:
        print(f"Processando áudio ID: {audio_id}")
        inputs = processor(audio_block, sampling_rate=SAMPLE_RATE, return_tensors="pt", task="transcribe", language="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move os tensores para a GPU

        generated_ids = model.generate(
            inputs["input_features"],
            max_length=50,
            num_beams=1,
            no_repeat_ngram_size=2,
        )
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Áudio ID: {audio_id} - Transcrição: {transcription.lower()}")
        return transcription.lower()
    except Exception as e:
        print(f"Erro durante a transcrição do áudio ID {audio_id}: {e}")
        return ""

def process_audio_queue():
    global current_command
    while True:
        audio_block, audio_id = audio_queue.get()
        transcription = transcribe_audio_block(audio_block, audio_id)
        
        with lock:
            if "left" in transcription:
                current_command = "left"
            elif "right" in transcription:
                current_command = "right"
            elif "run" in transcription:
                current_command += " run"
            elif "8" in transcription or "eight" in transcription:
                current_command += " coin"
            elif "stop" in transcription:
                current_command = ""
        
        audio_queue.task_done()

def main():
    print("Capturando áudio... Pressione Ctrl+C para interromper.")
    try:
        while True:
            audio_block = sd.rec(
                int(SAMPLE_RATE * BLOCK_DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32'
            )
            sd.wait()
            audio_block = np.squeeze(audio_block)
            audio_id = str(uuid.uuid4())
            audio_queue.put((audio_block, audio_id))
    except Exception as e:
        print(f"Erro no loop principal: {e}")

def release_all_keys():
    keyboard.release(Key.left)
    keyboard.release(Key.right)
    keyboard.release(Key.shift)
    keyboard.release(Key.down)

def control_game():
    global current_command
    try:
        while True:
            with lock:
                command = current_command

            release_all_keys()
            print(f"Comando atual: {command}")
            if "left" in command:
                keyboard.press(Key.left)
            elif "right" in command:
                keyboard.press(Key.right)
            elif command == "stop":
                release_all_keys()
            
            if "run" in command:
                keyboard.press(Key.shift)
            if "coin" in command:
                keyboard.press(Key.down)
                time.sleep(0.1)
            
            time.sleep(0.1)
    except Exception as e:
        print(f"Erro no loop secundário: {e}")

if __name__ == "__main__":
    try:
        # Configuração dos threads
        main_thread = threading.Thread(target=main, daemon=True)
        game_thread = threading.Thread(target=control_game, daemon=True)
        
        # Pool de threads para processamento de áudio
        workers = [threading.Thread(target=process_audio_queue, daemon=True) for _ in range(MAX_WORKERS)]
        
        main_thread.start()
        game_thread.start()
        for worker in workers:
            worker.start()

        while main_thread.is_alive() and game_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
    except Exception as e:
        print(f"Erro geral: {e}")
