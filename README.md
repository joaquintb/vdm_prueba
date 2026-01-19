# Prueba Técnica VDM Health
**Autor**: Joaquín Torres Bravo.

## Introducción
Este repositorio contiene la implementación completa de la prueba técnica para VDM Health, centrada en el uso de modelos multimodales y generativos aplicados al dominio de la imagen médica.

El objetivo de esta prueba es desarrollar un pipeline que integra: carga y preparación del dataset, etiquetado automático mediante un modelo CLIP-like, fine-tuning ligero con LoRA, y generación de imágenes sintéticas condicionadas por texto (Stable Diffusion + LCM).

## Estructura del Repositorio
* **`data/`**: carpeta destinada a datos locales (no versionados en Git).

  * **`subset/`**: subset exportado (imágenes + `metadata.jsonl`) para el fine-tuning LoRA.
  * **`.gitkeep`**: mantiene la carpeta `data/` en el repo aunque esté vacía.
  * **`pneumoniamnist.npz`**: archivo descargado del dataset (MedMNIST) para ejecución local.

* **`docker/`**

  * **`Dockerfile`**: imagen Docker basada en PyTorch con soporte CUDA (fallback a CPU) para ejecutar el pipeline completo.

* **`results/`**: outputs generados por el pipeline.

  * **`generated_images/`**: imágenes generadas (usando LCM + LoRA fine-tuned).
  * **`generated_images_no_fine_tuning/`**: generación base (LCM sin LoRA fine-tuned), usada como referencia.
  * **`lora/`**: artefactos del fine-tuning LoRA (pesos entrenados y/o checkpoints).
  * **`metrics/`**: métricas básicas para el etiquetado automático.
  * **`labeled_dataset.csv`**: CSV principal con `image_id`, `auto_label`, `ground_truth` y `confidence_score`.
  * **`pipeline_summary.json`**: resumen del run (rutas de artefactos y configuración/resultados principales).

* **`src/`**: código fuente del proyecto, organizado por módulos.

  * **`dataset/`**

    * **`__init__.py`**: define el paquete.
    * **`data_loading.py`**: carga del dataset y creación de DataLoaders por split.
    * **`prepare_and_label.py`**: prepara dataset, hace etiquetado automático y exporta subset para LoRA.
  * **`generation/`**

    * **`__init__.py`**: define el paquete.
    * **`lcm_generate.py`**: generación de imágenes con Stable Diffusion + LCM (+ LoRA fine-tuned opcional).
    * **`lora_fine_tuning.py`**: wrapper para lanzar fine-tuning LoRA usando el script oficial de Diffusers.
  * **`semantic/`**

    * **`__init__.py`**: define el paquete.
    * **`multimodal_recognition.py`**: inferencia CLIP-like (BiomedCLIP) para similitud texto–imagen y predicción de etiquetas.
  * **`pipeline.py`**: orquestación end-to-end del pipeline (labeling → LoRA → generación), con flags para saltar etapas.

* **`third_party/diffusers/`**

  * **`train_text_to_image_lora.py`**: script oficial de Diffusers para entrenar LoRA.

* **`tools/`**: utilidades auxiliares (no forman parte del pipeline principal).

  * **`colab_testing.ipynb`**: notebook usado para pruebas/ejecución en Colab (GPU).
  * **`smoke_test_imports.py`**: smoke test para verificar imports y dependencias del entorno.

* **`README.md`**: documentación del proyecto y guía de ejecución.

* **`.gitignore`**: exclusiones de Git (datasets, outputs pesados, caches, etc.).

* **`requirements.txt`**: dependencias completas para desarrollo/local.

* **`requirements-docker.txt`**: dependencias mínimas para ejecutar en Docker.
