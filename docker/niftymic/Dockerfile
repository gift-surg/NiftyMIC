#
# Building of Docker image with default monaifbs segmentation tool:
#   docker build --build-arg VERSION=v? -t renbem/niftymic:v? -t renbem/niftymic .
#
# If building with fetal_brain_seg as segmentation pipeline:
#   docker build --build-arg VERSION=v? --build-arg FETAL_SEG_TOOL=fetal_brain_seg -t renbem/niftymic:v? -t renbem/niftymic .

ARG VERSION=latest
ARG REPO=NiftyMIC
# default use monaifbs for segmentation. Define this arg as fetal_brain_seg to use previous app
# https://github.com/gift-surg/fetal_brain_seg.git
ARG FETAL_SEG_TOOL=monaifbs


# GUI with ITK-Snap does not work at the moment, unfortunately
ARG IMAGE=renbem/simplereg_dependencies:noitksnap

# -----------------------------------------------------------------------------
FROM $IMAGE as compile-image

ARG REPO
ARG VERSION
ARG FETAL_SEG_TOOL

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        git \
        && \
    rm -rf /var/lib/apt/lists/* 

# download NiftyMIC
RUN if [ "$VERSION" = "latest" ] ; then \
        git clone \
        https://github.com/gift-surg/${REPO}.git /app/${REPO} \
    ;else \
        git clone \
        --branch ${VERSION} \
        https://github.com/gift-surg/${REPO}.git /app/${REPO} \
    ;fi

# fetch MONAIfbs and download pretrained model for MONAIfbs
RUN if [ "$FETAL_SEG_TOOL" = "monaifbs" ] ; then \
        cd /app/${REPO} && \
        git submodule update --init && \
        # fetch the pretrained model
        cd /app && \
        pip install zenodo-get && \
        zenodo_get 10.5281/zenodo.4282679 && \
        tar xvf models.tar.gz && \
        mv models /app/${REPO}/MONAIfbs/monaifbs/ && \
        # remove the downloaded compressed file
        rm -r /app/models.tar.gz \
    ;fi

# download fetal_brain_seg if required (need to create an empty directory for following copy, line 105)
RUN mkdir /app/fetal_brain_seg
ADD https://github.com/taigw/Demic/archive/v0.1.tar.gz /app/Demic-0.1.tar.gz
RUN if [ "$FETAL_SEG_TOOL" = "fetal_brain_seg" ] ; then \
        git clone \
        https://github.com/gift-surg/fetal_brain_seg.git /app/fetal_brain_seg  && \
        cd /app && \
        tar xvf Demic-0.1.tar.gz && \
        mv Demic-0.1 /app/fetal_brain_seg/Demic && \
        # remove unecessary .git folders
        rm -r /app/fetal_brain_seg/.git* && \
        rm -r /app/fetal_brain_seg/Demic/.git* \
    ;fi

# remove unnecessary folders
RUN rm -r /app/${REPO}/.git*
RUN rm -r /app/Demic-0.1.tar.gz

# -----------------------------------------------------------------------------
FROM $IMAGE AS runtime-image

ARG REPO
ARG VERSION
ARG FETAL_SEG_TOOL

LABEL author="Michael Ebner"
LABEL email="michael.ebner@kcl.ac.uk"
LABEL title="$REPO"
LABEL version="$VERSION"
LABEL uri="https://github.com/gift-surg/${REPO}"

# install NiftyMIC with specific python library versions
COPY --from=compile-image /app/${REPO} /app/${REPO}
WORKDIR /app/${REPO}
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        nifti2dicom \
        && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
    matplotlib==3.1.1 \
    natsort==6.0.0 \
    nibabel==2.4.1 \
    nipype==1.2.0 \
    nose==1.3.7 \
    numpy==1.16.4 \
    pandas==0.25.0 \
    pydicom==1.3.0 \
    scikit_image==0.15.0 \
    scipy==1.3.0 \
    seaborn==0.9.0 \
    SimpleITK==1.2.4 \
    six==1.12.0 \
    pysitk==0.2.19 \
    simplereg==0.3.2 \
    nsol==0.1.14
# install monaifbs dependencies
RUN if [ "$FETAL_SEG_TOOL" = "monaifbs" ] ; then \
        pip install \
        torch==1.4.0 \
        torch-summary==1.4.3 \
        monai==0.3.0  \
        pyyaml==5.3.1 \
        pytorch-ignite==0.4.2 \
        tensorboard==2.3.0 \
    ;fi
# install packages for niftymic and monaifbs
RUN pip install -e .
RUN if [ "$FETAL_SEG_TOOL" = "monaifbs" ] ; then \
        pip install -e /app/${REPO}/MONAIfbs/ \
    ;fi

# prepare fetal_brain_seg with specific python library versions if required
COPY --from=compile-image /app/fetal_brain_seg /app/fetal_brain_seg
RUN if [ "$FETAL_SEG_TOOL" = "fetal_brain_seg" ] ; then \
        cd /app/fetal_brain_seg && \
        pip install \
        niftynet==0.2 \
        tensorflow==1.12.0 && \
        SITEDIR=$(python -m site --user-site) && \
        mkdir -p $SITEDIR && \
        echo /app/fetal_brain_seg > $SITEDIR/Demic.pth && \
        export FETAL_BRAIN_SEG=/app/fetal_brain_seg \
    ;else \
        rm -r /app/fetal_brain_seg \
    ;fi

# add Dockerfile to image
ADD Dockerfile /

WORKDIR /app

# use bash with color output
RUN echo 'alias ls="ls --color=auto"' >> ~/.bashrc
CMD bash