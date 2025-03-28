import os
import time

def monitor_gpu_nvidia_smi():
    while True:
        os.system('nvidia-smi')  # Ejecuta nvidia-smi
        time.sleep(1)  # Espera 1 segundo
        print("\n" * 2)  # Añade líneas en blanco para separar salidas

if __name__ == "__main__":
    try:
        monitor_gpu_nvidia_smi()
    except KeyboardInterrupt:
        print("Monitor detenido")
