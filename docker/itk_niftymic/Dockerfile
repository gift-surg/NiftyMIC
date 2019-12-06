#
# Building of Docker image:
#   docker build --build-arg VERSION=v? -t renbem/itk_niftymic:v? -t renbem/itk_niftymic .

ARG VERSION=latest
ARG REPO=ITK_NiftyMIC
ARG IMAGE=python:3.6-slim

# -----------------------------------------------------------------------------
FROM $IMAGE as compile-image

ARG REPO
ARG VERSION

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        git \
        && \
    rm -rf /var/lib/apt/lists/* 
RUN if [ "$VERSION" = "latest" ] ; then \
            git clone \
            https://github.com/gift-surg/${REPO}.git /code/${REPO}/${REPO} \
        ;else \
            git clone \
            --branch ${VERSION} \
            https://github.com/gift-surg/${REPO}.git /code/${REPO}/${REPO} \
        ;fi
RUN mkdir -p /code/${REPO}/${REPO}-build && \
    cd /code/${REPO}/${REPO}-build && \
    cmake \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_SHARED_LIBS=ON \
        -D BUILD_TESTING=OFF \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/code/${REPO}/${REPO}-install/itk \
        -D ITK_LEGACY_SILENT=ON \
        -D ITK_WRAP_covariant_vector_double=ON \
        -D ITK_WRAP_double=ON \
        -D ITK_WRAP_float=ON \
        -D ITK_WRAP_PYTHON=ON \
        -D ITK_WRAP_signed_char=ON \
        -D ITK_WRAP_signed_long=ON \
        -D ITK_WRAP_signed_short=ON \
        -D ITK_WRAP_unsigned_char=ON \
        -D ITK_WRAP_unsigned_long=ON \
        -D ITK_WRAP_unsigned_short=ON \
        -D ITK_WRAP_vector_double=ON \
        -D ITK_WRAP_vector_float=ON \
        -D Module_BridgeNumPy=ON \
        -D Module_ITKReview=ON \
        -D Module_SmoothingRecursiveYvvGaussianFilter=ON \
        -D WRAP_ITK_INSTALL_PREFIX=/code/${REPO}/${REPO}-install/wrapitk \
        /code/${REPO}/${REPO}
RUN cd /code/${REPO}/${REPO}-build && make -j 4
# RUN cd /code/${REPO}/${REPO}-build && make install

# remove unnecessary .git folders
RUN rm -r /code/${REPO}/${REPO}/.git*

# -----------------------------------------------------------------------------
FROM compile-image AS runtime-image
# FROM $IMAGE AS runtime-image

ARG REPO
ARG VERSION

LABEL author="Michael Ebner"
LABEL email="michael.ebner@kcl.ac.uk"
LABEL title="$REPO"
LABEL version="$VERSION"
LABEL uri="https://github.com/gift-surg/${REPO}"

WORKDIR /code/${REPO}

# add Dockerfile to image
ADD Dockerfile /

CMD bash

