podman run --rm -it \
    -v $(pwd)/scripts/python:/workspace:Z \
    secbert \
    bash