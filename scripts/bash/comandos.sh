podman run --rm -it \
    -v $(pwd)/scripts/python:/workspace:Z \
    -v $(pwd)/datasets:/data:Z \
    secbert \
    bash

python etl.py --input logs.csv --output logs_clean.csv

