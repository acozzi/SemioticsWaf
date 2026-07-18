podman build -t latex-fedora-local .
podman run --rm -it -v "$PWD:/workspace" -w /workspace/FINAL localhost/latex-fedora-local latexmk -pdf tpfi.tex
