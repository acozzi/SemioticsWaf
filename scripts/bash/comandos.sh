podman run --rm -it \
    -v $(pwd)/scripts/python:/workspace:Z \
    -v $(pwd)/datasets:/data:Z \
    -e HF_TOKEN="${HF_TOKEN}" \
    -w /workspace \
    secbert \
    bash

python etl.py --input logs.csv --output logs_clean.csv

podman run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi
podman run --rm --security-opt=label=disable --device nvidia.com/gpu=all ubuntu nvidia-smi


podman run --rm -it \
    -v $(pwd)/scripts/python:/workspace:Z \
    -v $(pwd)/datasets:/data:Z \
    --device nvidia.com/gpu=all \
    --memory=8g \
    --memory-swap=8g \
    -w /workspace \
    localhost/secbert:0.1cuda \
    bash

python extractor.py --batch-size 64 --group-size 50

podman run --rm -it \
    -v $(pwd)/scripts/python:/workspace:Z \
    -v $(pwd)/datasets:/data:Z \
    --device nvidia.com/gpu=all \
    --memory=10g \
    --memory-swap=16g \
    -w /workspace \
    localhost/secbert:0.1cuda \
    bash