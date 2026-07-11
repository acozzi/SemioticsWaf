podman run --rm -it \
    -v $(pwd)/scripts/python:/workspace \
    -v $(pwd)/datasets:/data \
    secbert \
    bash
