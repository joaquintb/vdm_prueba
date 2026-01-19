# Prueba Técnica VDM Health
**Autor:** Joaquín Torres Bravo

## Introducción
Este repositorio contiene la implementación completa de la prueba técnica para VDM Health, centrada en el uso de modelos multimodales y generativos aplicados al dominio de la imagen médica.

El objetivo de esta prueba es desarrollar un pipeline que integra: carga y preparación del dataset, etiquetado automático mediante un modelo CLIP-like, fine-tuning ligero con LoRA, y generación de imágenes sintéticas condicionadas por texto (Stable Diffusion + LCM).

Repositorio en GitHub: https://github.com/joaquintb/vdm_prueba

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
  * **`pipeline_summary.json`**: resumen del run (rutas de artefactos y resultados/configuración principales).

* **`src/`**: código fuente del proyecto, organizado por módulos.
  * **`dataset/`**
    * **`__init__.py`**: define el paquete.
    * **`data_loading.py`**: carga del dataset y creación de DataLoaders por split.
    * **`prepare_and_label.py`**: prepara el dataset, realiza etiquetado automático y exporta el subset para LoRA.
  * **`generation/`**
    * **`__init__.py`**: define el paquete.
    * **`lcm_generate.py`**: generación con Stable Diffusion + LCM (+ LoRA fine-tuned opcional).
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
  * Generación de **`results/labeled_dataset.csv`** con `image_id`, `auto_label`, `ground_truth` y `confidence_score`.
  * Exportación de un subset balanceado (imágenes + `metadata.jsonl`) para fine-tuning LoRA.

* **Parte 2 — Reconocimiento Semántico Texto–Imagen**
  * **Modelo:** BiomedCLIP.
  * **Uso:** similitud entre embeddings de imagen y prompts clínicos → etiqueta (`auto_label`) + confianza.

* **Parte 3 — Generación de Imágenes con LCM**
  * **Modelo base:** Stable Diffusion v1.5.
  * **Aceleración:** adaptador LCM LoRA + LCMScheduler para generar en pocos pasos.
  * **Output:** ≥10 imágenes en `results/generated_images/`.

* **Parte 4 — Pipeline Integrado + Fine-tuning ligero**
  * **Fine-tuning:** LoRA (entrenamiento ligero con Diffusers) sobre el subset exportado.
  * **Integración:** el LoRA entrenado se carga en el generador y se utiliza para la generación final.
  * Pipeline secuencial y reproducible (disk-based), con artefactos guardados en `results/` y ejecución configurable por flags.

## Decisiones Técnicas Principales

### Elección del dataset: PneumoniaMNIST (MedMNIST)
Se eligió **PneumoniaMNIST** (benchmark **MedMNIST**) por ser un dataset muy conocido y ampliamente utilizado en la comunidad académica de visión por computador aplicada a imagen médica. MedMNIST está diseñado para prototipado rápido y evaluación de pipelines, ofreciendo datasets estandarizados y fáciles de integrar.

PneumoniaMNIST resulta especialmente adecuado porque:
* Contiene imágenes médicas reales (radiografías de tórax).
* Proporciona labels fiables y bien definidos.
* Es un problema binario (normal vs. neumonía), lo que reduce la ambigüedad semántica.
* Simplifica la definición de prompts textuales y la evaluación de resultados.

MedMNIST: https://medmnist.com/

### Modelo CLIP: BiomedCLIP
Se eligió **BiomedCLIP** por su simplicidad de integración y adecuación al dominio médico. Mantiene la misma interfaz conceptual que CLIP (embeddings de imagen y texto + cálculo de similitud), lo que permite implementar el reconocimiento semántico basado en prompts de forma rápida y limpia.

Además, el modelo está disponible en Hugging Face, con pesos, documentación clara y licencia MIT, lo que facilita su uso en un entorno reproducible (Docker + dependencias). Al estar preentrenado en datos biomédicos multimodales (incluyendo radiografías), ofrece una alineación semántica mejor que un CLIP generalista sin necesidad de entrenamiento adicional.

Modelo: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

### Generación de Imágenes: SD v1.5 + LCM + LoRA
Para la generación de imágenes sintéticas se utiliza **Stable Diffusion v1.5** como modelo base por ser el backend más estable y ampliamente soportado en **Diffusers**, garantizando compatibilidad y facilidad de integración.

Sobre este modelo se incorpora **LCM** mediante el adaptador `lcm-lora-sdv1-5`, diseñado para SD v1.5. Esto permite acelerar la generación (2–8 pasos frente a decenas en difusión estándar), reduciendo el coste computacional y facilitando la ejecución en entornos con recursos limitados.

Finalmente, se aplica **LoRA** como mecanismo de fine-tuning ligero para adaptar el modelo al dominio de radiografías de tórax sin modificar ni almacenar el modelo completo.

## Resultados

### Etiquetado Automático

#### Métricas por split
| Split       | Nº muestras | Accuracy | TN | FP   | FN | TP   |
| ----------- | -----------:| -------: | -:| ----:| -:| ----:|
| Train       | 4708        | 0.742    | 0 | 1214 | 0 | 3494 |
| Val         | 524         | 0.742    | 0 | 135  | 0 | 389  |
| Test        | 624         | 0.625    | 0 | 234  | 0 | 390  |
| **Overall** | **5856**    | **0.730**| **0** | **1583** | **0** | **4273** |

Los resultados muestran una accuracy global cercana al 73%, con valores consistentes entre train y validación, y una caída moderada en test.

#### Limitaciones y mejoras
Las métricas no son especialmente altas, lo cual es esperable dado que **BiomedCLIP se ha utilizado en modo zero-shot**, sin fine-tuning sobre el dataset específico. Además, aunque BiomedCLIP está preentrenado en datos biomédicos multimodales, su entrenamiento no está restringido a radiografías de tórax. Un modelo más especializado en chest X-rays o un fine-tuning ligero sobre datasets específicos de radiografía de tórax podría mejorar el rendimiento.

Aun así, el rendimiento es razonablemente bueno para un escenario de prototipado rápido y suficiente para cumplir el objetivo principal de la prueba. Para la fase de generación de imágenes se emplean etiquetas **ground truth** para evitar propagar errores del etiquetado automático y estabilizar los prompts.

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

En la comparación visual, la generación con LoRA tiende a producir imágenes más alineadas con el estilo de radiografía (tonos e intensidad más plausibles), aunque con suavizado y pérdida de detalle anatómico. Sin fine-tuning, la generación tiende a estructuras más genéricas (ilustración/esqueleto), lo que sugiere menor alineación del modelo base con el dominio de X-ray.

Esto es coherente con el enfoque adoptado: el LoRA introduce una adaptación ligera al dominio con un subconjunto pequeño y entrenamiento limitado (200 pasos), suficiente para orientar el modelo hacia el tipo de imagen deseado, pero no para capturar patrones anatómicos finos o patología compleja.

#### Limitaciones y mejoras 
* Entrenamiento LoRA muy ligero: pocos pasos y subset reducido limitan el nivel de detalle.
* Resolución baja (256×256): reduce el coste computacional, pero limita realismo clínico.
* Modelo base genérico (SD v1.5): no está especializado en imagen médica.
* Prompts simples: prompts más estructurados podrían guiar mejor la generación.

## Desarrollo
* **Entorno local:** Windows 11 sin GPU. Esto limitó la ejecución local de modelos pesados y motivó el uso de un entorno con GPU para fases costosas.
* **Ejecución con GPU:** Google Colab para validar BiomedCLIP, generación con LCM y fine-tuning LoRA, además de comprobar el pipeline completo.
  * Notebook auxiliar: `tools/colab_testing.ipynb`
* **Edición de código:** VS Code.
* **Control de versiones:** Git.
* **Reproducibilidad:** Docker para encapsular dependencias y permitir ejecución con fallback a CPU.

> **Nota sobre recursos computacionales**  
> Los parámetros con mayor impacto computacional (batch size, pasos de inferencia o iteraciones de fine-tuning) se han ajustado para ejecutarse en una GPU NVIDIA T4 de Google Colab. En otros entornos deben adaptarse a los recursos disponibles.

## Docker
### Instrucciones de Ejecución

#### 1) Build de la imagen

Desde la raíz del repositorio:

```bash
docker build -t vdm-pipeline -f docker/Dockerfile .
```

---

#### 2) Run básico (ejecuta el pipeline completo)

```bash
docker run --rm vdm-pipeline
```

Por defecto, el contenedor ejecuta todo el pipeline: `python -m src.pipeline --force` (a sobreescribir con los comandos de abajo).

---

#### 3) Persistir resultados en tu máquina (recomendado)

Para que `results/` y `data/` se guarden fuera del contenedor (en tu PC), monta volúmenes:

**PowerShell (Windows):**

```powershell
docker run --rm `
  -v ${PWD}\results:/app/results `
  -v ${PWD}\data:/app/data `
  vdm-pipeline
```

---

#### 4) Ejecutar solo parte del pipeline (flags)

El script `src/pipeline.py` soporta:

* `--skip_labeling`: salta preparación/etiquetado
* `--skip_lora`: salta fine-tuning LoRA
* `--skip_generation`: salta generación
* `--force`: recalcula desde cero (ignora artefactos existentes)

Ejemplos:

**Recalcular todo desde cero**

```powershell
docker run --rm `
  -v ${PWD}\results:/app/results `
  -v ${PWD}\data:/app/data `
  vdm-pipeline python -m src.pipeline --force
```

**Ejecutar pipeline pero sin LoRA**

```powershell
docker run --rm `
  -v ${PWD}\results:/app/results `
  -v ${PWD}\data:/app/data `
  vdm-pipeline python -m src.pipeline --skip_lora
```

**Solo etiquetado (sin LoRA ni generación)**

```powershell
docker run --rm `
  -v ${PWD}\results:/app/results `
  -v ${PWD}\data:/app/data `
  vdm-pipeline python -m src.pipeline --skip_lora --skip_generation
```

**Solo generación (asumiendo que ya existe LoRA/pesos y/o artefactos previos)**

```powershell
docker run --rm `
  -v ${PWD}\results:/app/results `
  -v ${PWD}\data:/app/data `
  vdm-pipeline python -m src.pipeline --skip_labeling
```

### Smoke Test
Dado que la ejecución completa del pipeline dentro de Docker puede ser costosa sin GPU, se incluye un smoke test que importa los módulos y scripts principales para verificar que el entorno y las dependencias están correctamente configurados. Esto valida el funcionamiento básico del contenedor sin necesidad de ejecutar el pipeline completo (ya validado en Colab).

### Requirements
Se utiliza un archivo `requirements-docker.txt` reducido para minimizar el tiempo de build y evitar dependencias innecesarias dentro de Docker. Las librerías incluidas son las estrictamente necesarias para ejecutar el pipeline. PyTorch no se fija en este archivo, ya que su versión se define en la imagen base de Docker, evitando conflictos de dependencias.



