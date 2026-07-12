podman run --rm -it \
    -v $(pwd)/scripts/python:/workspace:Z \
    -v $(pwd)/datasets:/data:Z \
    secbert \
    bash
