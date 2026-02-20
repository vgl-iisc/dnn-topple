FROM pytorch/pytorch
ADD *.py /workspace/
WORKDIR /workspace/
RUN pip install torchvision torchaudio tensorboard numpy tqdm scipy scikit-learn PyYAML

CMD ["python", "run_train_inference.py", "--config", "/runs/resnet50.yaml", "-o", "/logs/resnet50.log"]