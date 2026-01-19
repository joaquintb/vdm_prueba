docker build -f docker\Dockerfile -t vdm-pipeline .

docker run --rm `
  -v ${PWD}\data:/app/data `
  vdm-pipeline --force --skip-lora

docker images | findstr vdm-pipeline


docker run --rm vdm-pipeline python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"

docker run --rm `
  -v ${PWD}:/app `
  vdm-pipeline