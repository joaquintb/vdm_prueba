# vdm_prueba


## Ejecución del pipeline

El pipeline completo se ejecuta desde la raíz del proyecto con:

```bash
python -m src.pipeline
```

Este comando ejecuta secuencialmente:

1. Preparación y etiquetado automático del dataset.
2. Fine-tuning ligero (LoRA) sobre un subset etiquetado.
3. Generación de imágenes médicas condicionadas por texto.

### Opciones disponibles

* **`--skip_labeling`**
  Omite la fase de preparación y etiquetado si los resultados ya existen.

* **`--skip_lora`**
  Omite el fine-tuning LoRA y reutiliza pesos previamente entrenados.

* **`--skip_generation`**
  Omite la generación de imágenes.

* **`--force`**
  Fuerza la reejecución de una fase aunque sus resultados ya existan.

* **`--num_images N`**
  Número de imágenes a generar en la fase de generación (por defecto: 10).

### Ejemplos

Ejecutar todo el pipeline:

```bash
python -m src.pipeline
```

Generar imágenes usando el LoRA ya entrenado:

```bash
python -m src.pipeline --skip_labeling --skip_lora
```

Reentrenar LoRA sin repetir el etiquetado:

```bash
python -m src.pipeline --skip_labeling --force
```
