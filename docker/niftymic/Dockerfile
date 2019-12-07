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

# download fetal_brain_seg
RUN git clone \
    https://github.com/gift-surg/fetal_brain_seg.git /app/fetal_brain_seg
ADD https://github.com/taigw/Demic/archive/v0.1.tar.gz /app/Demic-0.1.tar.gz
RUN cd /app && \
    tar xvf Demic-0.1.tar.gz && \
    mv Demic-0.1 /app/fetal_brain_seg/Demic

# download NiftyMIC
RUN if [ "$VERSION" = "latest" ] ; then \
        git clone \
        https://github.com/gift-surg/${REPO}.git /app/${REPO} \
    ;else \
        git clone \
        --branch ${VERSION} \
        https://github.com/gift-surg/${REPO}.git /app/${REPO} \
    ;fi

# remove unnecessary .git folders
RUN rm -r /app/${REPO}/.git*
RUN rm -r /app/fetal_brain_seg/.git*
RUN rm -r /app/fetal_brain_seg/Demic/.git*

# -----------------------------------------------------------------------------
FROM $IMAGE AS runtime-image

ARG REPO
ARG VERSION

LABEL author="Michael Ebner"
LABEL email="michael.ebner@kcl.ac.uk"
LABEL title="$REPO"
LABEL version="$VERSION"
LABEL uri="https://github.com/gift-surg/${REPO}"

# prepare fetal_brain_seg
COPY --from=compile-image /app/fetal_brain_seg /app/fetal_brain_seg
WORKDIR /app/fetal_brain_seg
RUN pip install \
    nibabel==2.3.3 \
    niftynet==0.2 \
    numpy==1.15.4 \
    scikit_image==0.14.1 \
    scipy==1.1.0 \
    SimpleITK==1.2.4 \
    tensorflow==1.12.0
RUN SITEDIR=$(python -m site --user-site) && \
    mkdir -p $SITEDIR && \
    echo /app/fetal_brain_seg > $SITEDIR/Demic.pth
ENV FETAL_BRAIN_SEG=/app/fetal_brain_seg

# install NiftyMIC
COPY --from=compile-image /app/${REPO} /app/${REPO}
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libblas-dev \
        nifti2dicom \
        libboost-program-options-dev \
        cmake \
        && \
    rm -rf /var/lib/apt/lists/* 

WORKDIR /app/${REPO}
RUN pip install six
RUN pip install -e .

# add Dockerfile to image
ADD Dockerfile /

WORKDIR /app
CMD bash