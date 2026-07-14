FROM fedora:44

RUN dnf install -y \
    texlive-scheme-basic \
    texlive-collection-latexextra \
    texlive-collection-fontsrecommended \
    texlive-collection-langspanish \
    texlive-helvetic \
    texlive-biblatex \
    biber \
    latexmk \
    make \
    git \
    which \
    && dnf clean all

WORKDIR /workspace
