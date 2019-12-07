#
# Building of Docker image:
#   docker build --build-arg VERSION=v? -t renbem/niftymic:v? -t renbem/niftymic .

ARG VERSION=latest
ARG REPO=NiftyMIC

# GUI with ITK-Snap does not work at the moment, unfortunately
ARG IMAGE=renbem/simplereg_dependencies:noitksnap

# -----------------------------------------------------------------------------
FROM $IMAGE as compile-image

ARG REPO
ARG VERSION

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        git \
        && \
    rm -rf /var/lib/apt/lists/* 

RUN if [ "$VERSION" = "latest" ] ; then \
        git clone \
        https://github.com/gift-surg/${REPO}.git /app/${REPO} \
    ;else \
        git clone \
        --branch ${VERSION} \
        https://github.com/gift-surg/${REPO}.git /app/${REPO} \
    ;fi

WORKDIR /app/${REPO}

# remove unnecessary .git folders
RUN rm -r /app/${REPO}/.git*

# -----------------------------------------------------------------------------
FROM $IMAGE AS runtime-image

ARG REPO
ARG VERSION

LABEL author="Michael Ebner"
LABEL email="michael.ebner@kcl.ac.uk"
LABEL title="$REPO"
LABEL version="$VERSION"
LABEL uri="https://github.com/gift-surg/${REPO}"

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libblas-dev \
        nifti2dicom \
        libboost-program-options-dev \
        cmake \
        && \
    rm -rf /var/lib/apt/lists/* 

COPY --from=compile-image /app/${REPO} /app/${REPO}

WORKDIR /app/${REPO}
RUN pip install six
RUN pip install -e .

# add Dockerfile to image
ADD Dockerfile /

WORKDIR /app
CMD bash