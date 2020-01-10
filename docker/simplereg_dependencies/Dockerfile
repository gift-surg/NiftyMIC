#
# Building of Docker image:
#   docker build -t renbem/simplereg_dependencies .
#   
# Run with GUI (however, does not work unfortunately):
#   xhost +local:docker  # needed only the first time
#   docker run --rm -ti --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" renbem/simplereg_dependencies

ARG VERSION=noitksnap
ARG REPO=SimpleReg-dependencies
ARG IMAGE=renbem/itk_niftymic:v4.13.1-niftymic-v1

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
FROM $IMAGE as compile-image-itksnap
# itksnap GUI not working unfortunately

# convert3D
ADD c3d-1.0.0-Linux-x86_64.tar.gz /code/
RUN mv /code/c3d-1.0.0-Linux-x86_64 /code/c3d

# itksnap with QT4 opens but GUI has issues then
# a)
ADD itksnap-3.8.0-20190612-Linux-x86_64-qt4.tar.gz /code/
RUN mv /code/itksnap-3.8.0-20190612-Linux-gcc64-qt4 /code/itksnap
# b)
# ADD itksnap-nightly-master-Linux-x86_64-qt4.tar.gz /code/
# RUN mv /code/itksnap-3.6.0-20170401-Linux-x86_64-qt4 /code/itksnap

# versions do not work: after 'itksnap', the following error:
# 
# [7]: The last reference on a connection was dropped without closing the connection. This is a bug in an application. See dbus_connection_unref() documentation for details.
# Most likely, the application was supposed to call dbus_connection_close(), since this is a private connection.
# D-Bus not built with -rdynamic so unable to print a backtrace
# Aborted (core dumped)
# 
# a)
# ADD itksnap-3.8.0-20190612-Linux-x86_64.tar.gz /code/
# RUN mv /code/itksnap-3.8.0-20190612-Linux-gcc64 /code/itksnap
# b)
# ADD itksnap-nightly-master-Linux-x86_64.tar.gz /code/
# RUN mv /code/itksnap-3.6.0-20170401-Linux-x86_64 /code/itksnap


# -----------------------------------------------------------------------------
FROM $IMAGE AS runtime-image

ARG REPO
ARG VERSION

LABEL author="Michael Ebner"
LABEL email="michael.ebner@kcl.ac.uk"
LABEL title="$REPO"
LABEL version="$VERSION"
LABEL uri="https://github.com/gift-surg/SimpleReg/wiki/simplereg-dependencies"

# copy compiled FSL files and link associated binaries
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

# copy compiled NiftyReg files and link associated binaries
COPY --from=compile-image-niftyreg /usr/share/niftyreg /usr/share/niftyreg
ENV PATH="/usr/share/niftyreg/bin:$PATH"

# copy compiled itksnap files
COPY --from=compile-image-itksnap /code/c3d/bin /usr/local/bin
COPY --from=compile-image-itksnap /code/c3d/lib /usr/local/lib
COPY --from=compile-image-itksnap /code/c3d/share /usr/local/share

COPY --from=compile-image-itksnap /code/itksnap/bin /usr/local/bin
COPY --from=compile-image-itksnap /code/itksnap/lib /usr/local/lib

# to make use of itksnap GUI within docker (in principle;
# but errors/problems as above; thus, deactivated for now)
# RUN apt-get update && \
#     apt-get install -y \
#         wget \
#         libglu1 \
#         libcurl4-openssl-dev \
#         libsm6 \
#         libxt6 \
#         libfreetype6 \
#         libxrender1 \
#         libfontconfig1 \
#         libglib2.0-0 \
#         libqt4-dev \
#         libgtk2.0-dev \
#         curl \
#         libgtk2.0 \
#         qt5dxcb-plugin \
#         && \
#     rm -rf /var/lib/apt/lists/*
# RUN apt-get update && \
#     apt-get install -y \
#         libxcb1 libxcb1-dev \
#         libx11-dev \
#         libgl1-mesa-dev \
#         libxt-dev libxft-dev \
#         && \
#     rm -rf /var/lib/apt/lists/*
# ADD libpng12-0_1.2.54-1ubuntu1.1_amd64.deb /code/
# RUN dpkg -i /code/libpng12-0_1.2.54-1ubuntu1.1_amd64.deb

# add Dockerfile to image
ADD Dockerfile /

# use bash with color output
RUN echo 'alias ls="ls --color=auto"' >> ~/.bashrc
CMD bash