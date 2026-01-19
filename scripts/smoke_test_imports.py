def main():
    import torch
    import diffusers
    import transformers
    import open_clip
    import peft
    import medmnist
    import accelerate
    import safetensors

    import src.pipeline
    import src.dataset.data_loading
    import src.dataset.prepare_and_label
    import src.semantic.multimodal_recognition
    import src.generation.lcm_generate
    import src.generation.lora_fine_tuning

    print("✅ Import smoke test passed")
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

if __name__ == "__main__":
    main()
