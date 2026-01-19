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

## Resumen de la Implementación y Modelos Utilizados
* **Dataset**

  * **PneumoniaMNIST (MedMNIST)**: radiografías de tórax (splits `train/val/test`) usadas como base para todo el pipeline.

* **Parte 1 — Preparación + Etiquetado Automático**

  * Carga del dataset y generación de **`results/labeled_dataset.csv`** con: `image_id`, `auto_label`, `ground_truth` y `confidence_score`.
  * Export de un **subset balanceado** (imágenes + `metadata.jsonl`) para fine-tuning LoRA.

* **Parte 2 — Reconocimiento Semántico Texto–Imagen**

  * **Modelo:** **BiomedCLIP** (CLIP-like adaptado a dominio médico).
  * **Uso:** similitud entre embeddings de imagen y prompts clínicos → etiqueta predicha (`auto_label`) + confianza.

* **Parte 3 — Generación de Imágenes con LCM**

  * **Modelo base:** **Stable Diffusion v1.5**.
  * **Aceleración:** **LCM LoRA adapter** (Latent Consistency) + **LCMScheduler** para generar en pocos pasos.
  * **Output:** ≥10 imágenes en **`results/generated_images/`**.

* **Parte 4 — Pipeline Integrado + Fine-tuning ligero**

  * **Fine-tuning:** **LoRA** (entrenamiento ligero con Diffusers) sobre el subset exportado.
  * **Integración:** el LoRA entrenado se carga en el generador y se utiliza para la generación final.
  * Pipeline **secuencial y reproducible (disk-based)**, con artefactos guardados en `results/` y ejecución configurable por flags.

## Decisiones Técnicas Principales

### Elección del dataset: PneumoniaMNIST (MedMNIST)
Se eligió **PneumoniaMNIST**, perteneciente al benchmark **MedMNIST**, por ser un dataset **muy conocido y ampliamente utilizado en la comunidad académica de visión por computador aplicada a imagen médica**. MedMNIST está específicamente diseñado para **prototipado rápido y evaluación de pipelines**, ofreciendo datasets estandarizados, bien documentados y fáciles de integrar.

En concreto, PneumoniaMNIST resulta especialmente adecuado porque:

* Contiene **imágenes médicas reales** (radiografías de tórax).
* Proporciona **labels fiables y bien definidos**.
* Es un problema **binario (normal vs. neumonía)**, lo que **reduce la ambigüedad semántica**.
* Simplifica la **definición de prompts textuales** y la evaluación de resultados.

Esta combinación permite centrarse en el diseño correcto del pipeline sin introducir complejidad innecesaria derivada del dataset.

Datasets MNIST: [https://medmnist.com/](https://medmnist.com/)

### Modelo CLIP: BiomedCLIP

Se eligió **BiomedCLIP** por su **simplicidad de integración y adecuación al dominio médico**. Mantiene la misma interfaz conceptual que CLIP (embeddings de imagen y texto + cálculo de similitud), lo que permite implementar el reconocimiento semántico basado en *prompts* de forma rápida y limpia.

Además, el modelo está disponible públicamente en Hugging Face, con pesos, documentación clara y licencia MIT, lo que facilita su uso en un entorno reproducible (Docker + `requirements.txt`). Al estar **preentrenado específicamente en datos biomédicos**, incluyendo radiografías, ofrece una mejor alineación semántica que un CLIP generalista sin necesidad de entrenamiento adicional.

Modelo:
[https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

### Generación de Imágenes: SD v1.5 + LCM + LoRA
Para la generación de imágenes médicas sintéticas se ha utilizado **Stable Diffusion v1.5** como modelo base, principalmente por ser el *backend* más estable y ampliamente soportado dentro de la librería **Diffusers**. Esta elección garantiza compatibilidad, documentación abundante y facilidad de integración.

Sobre este modelo se incorpora **LCM (Latent Consistency Models)** mediante el adaptador **`lcm-lora-sdv1-5`**, diseñado específicamente para SD v1.5. Este adaptador permite **acelerar drásticamente la generación** (2–8 pasos de inferencia frente a decenas en difusión estándar), reduciendo el coste computacional y haciendo viable la ejecución incluso en entornos con recursos limitados.

Finalmente, se aplica **LoRA** como mecanismo de *fine-tuning ligero*, permitiendo adaptar el modelo al dominio de radiografías de tórax sin modificar ni almacenar el modelo completo. Esto mantiene el entrenamiento simple, eficiente y fácilmente reutilizable dentro del pipeline.

## Resultados
### Etiquetado Automático

#### **Métricas por Split**
| Split       | Nº muestras | Accuracy  | TN    | FP        | FN    | TP        |
| ----------- | ----------- | --------- | ----- | --------- | ----- | --------- |
| Train       | 4 708       | 0.742     | 0     | 1 214     | 0     | 3 494     |
| Val         | 524         | 0.742     | 0     | 135       | 0     | 389       |
| Test        | 624         | 0.625     | 0     | 234       | 0     | 390       |
| **Overall** | **5 856**   | **0.730** | **0** | **1 583** | **0** | **4 273** |

Los resultados de etiquetado automático muestran una **accuracy global cercana al 73 %**, con valores consistentes entre *train* y *validation*, y una caída moderada en *test*. 

#### **Limitaciones y Mejoras**
Las métricas no son especialmente altas, lo cual es esperable dado que **BiomedCLIP se ha utilizado en modo *zero-shot***, sin ningún tipo de fine-tuning sobre el dataset específico.
Además, aunque BiomedCLIP está **preentrenado en datos biomédicos multimodales**, su entrenamiento no está restringido a radiografías de tórax. Un **modelo más especializado en chest X-rays** o un **fine-tuning ligero** sobre datasets específicos de radiografía de tórax podría mejorar significativamente el rendimiento en esta tarea.

Aun así, el rendimiento es razonablemente bueno para un escenario de prototipado rápido y suficiente para cumplir el objetivo principal de la prueba. En cualquier caso, **para la fase de generación de imágenes se emplean las etiquetas *ground truth***, con el fin de evitar la propagación de errores derivados del etiquetado automático y garantizar mayor estabilidad semántica en los prompts utilizados.

### Generación de imágenes
<table>
  <tr>
    <td align="center">
      <img src="results/generated_images/gen_021.png" width="250"/>
      <br/>
      <em>Imagen generada con LoRA fine-tuning</em>
    </td>
    <td align="center">
      <img src="results/generated_images_no_fine_tuning/gen_000.png" width="250"/>
      <br/>
      <em>Imagen generada sin fine-tuning</em>
    </td>
  </tr>
</table>

En la comparación visual se observa que la generación **con LoRA fine-tuning** produce imágenes con una distribución de intensidades y un aspecto más cercano al dominio de radiografías de tórax, aunque todavía con un nivel elevado de suavizado y pérdida de detalle anatómico. En cambio, la generación **sin fine-tuning** tiende a producir estructuras genéricas tipo esqueleto o ilustración médica, lo que indica que el modelo base no está correctamente alineado con el dominio de X-ray clínico.

Este resultado es coherente con el enfoque adoptado: el LoRA introduce una adaptación ligera al dominio a partir de un subconjunto pequeño y con entrenamiento muy limitado (200 pasos), suficiente para orientar el modelo hacia el tipo de imagen deseado, pero no para capturar detalles finos o patrones patológicos complejos. Aun así, el efecto del fine-tuning es visible y valida la integración correcta del pipeline de generación condicionada.


#### **Limitaciones y Mejoras**

* **Entrenamiento LoRA muy ligero**: el número de pasos (`max_train_steps = 200`) y el tamaño reducido del dataset limitan la capacidad del modelo para aprender estructuras anatómicas detalladas. Aumentar ambos mejoraría la calidad visual.

* **Resolución baja (256×256)**: necesaria para mantener bajo el coste computacional, pero insuficiente para radiografías médicas realistas. Un entrenamiento a mayor resolución (512×512) permitiría capturar mejor la anatomía pulmonar.

* **Modelo base genérico (SD v1.5)**: aunque ampliamente soportado, no está especializado en imagen médica. Partir de un modelo base entrenado específicamente en radiografías podría mejorar notablemente los resultados.

* **Uso de prompts simples**: los textos empleados son deliberadamente genéricos. Prompts más ricos o estructurados (por ejemplo, describiendo localización y tipo de opacidad) podrían guiar mejor la generación.

A pesar de estas limitaciones, los resultados confirman que el pipeline completo funciona correctamente de extremo a extremo.

## Desarrollo

- La princpial limitacion ha sido el acceso a GPU para poder colab was used to test each component of the pipeline individually and run the whole pipeline 
- vs code
- git
- Uso de inteligencia artificial

## Instrucciones de Ejecución con Docker
...

## Compatibilidad con GPU
...

