#
# Building of Docker image:
#   docker build -t renbem/simplereg_dependencies .

ARG REPO=SimpleReg-dependencies
ARG IMAGE=renbem/itk_niftymic:v0.1

# -----------------------------------------------------------------------------
FROM $IMAGE as compile-image-fsl

ARG REPO
ARG VERSION

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        wget \
        && \
    rm -rf /var/lib/apt/lists/* 

# install FSL
RUN wget -O- http://neuro.debian.net/lists/stretch.au.full | \
    tee /etc/apt/sources.list.d/neurodebian.sources.list
RUN apt-key adv --recv-keys --keyserver \
    hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
        apt-get install -y fsl-core \
        && \
    rm -rf /var/lib/apt/lists/* 

# -----------------------------------------------------------------------------
FROM $IMAGE as compile-image-niftyreg

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        git \
        && \
    rm -rf /var/lib/apt/lists/* 

# install NiftyReg
RUN git clone https://github.com/KCL-BMEIS/niftyreg.git /code/niftyreg
RUN mkdir -p /code/niftyreg-build && \
    mkdir -p /usr/share/niftyreg
RUN cd /code/niftyreg-build && \
    cmake \
        -D CMAKE_INSTALL_PREFIX=/usr/share/niftyreg \
        /code/niftyreg
RUN cd /code/niftyreg-build && make -j4
RUN cd /code/niftyreg-build && make install

# -----------------------------------------------------------------------------
FROM $IMAGE AS runtime-image

ARG REPO
ARG VERSION

LABEL author="Michael Ebner"
LABEL email="michael.ebner@kcl.ac.uk"
LABEL title="$REPO"

# copy compiled FSL data and link associated binaries
COPY --from=compile-image-fsl /etc/fsl /etc/fsl
COPY --from=compile-image-fsl /usr/lib /usr/lib
COPY --from=compile-image-fsl /usr/share/fsl /usr/share/fsl
COPY --from=compile-image-fsl /usr/bin/fsl5.0-* /usr/bin/
RUN ln -s /usr/bin/fsl5.0-flirt /usr/local/bin/flirt && \
    ln -s /usr/bin/fsl5.0-fslhd /usr/local/bin/fslhd && \
    ln -s /usr/bin/fsl5.0-fslmodhd /usr/local/bin/fslmodhd && \
    ln -s /usr/bin/fsl5.0-fslorient /usr/local/bin/fslorient && \
    ln -s /usr/bin/fsl5.0-fslreorient2std /usr/local/bin/fslreorient2std && \
    ln -s /usr/bin/fsl5.0-fslswapdim /usr/local/bin/fslswapdim && \
    ln -s /usr/bin/fsl5.0-bet /usr/local/bin/bet

# copy compiled NiftyReg data and link associated binaries
COPY --from=compile-image-niftyreg /usr/share/niftyreg /usr/share/niftyreg
ENV PATH="/usr/share/niftyreg/bin:$PATH"

# add Dockerfile to image
ADD Dockerfile /

CMD bash