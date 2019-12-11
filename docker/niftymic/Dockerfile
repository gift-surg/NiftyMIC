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

# download NiftyMIC
RUN if [ "$VERSION" = "latest" ] ; then \
        git clone \
        https://github.com/gift-surg/${REPO}.git /app/${REPO} \
    ;else \
        git clone \
        --branch ${VERSION} \
        https://github.com/gift-surg/${REPO}.git /app/${REPO} \
    ;fi

# download fetal_brain_seg
RUN git clone \
    https://github.com/gift-surg/fetal_brain_seg.git /app/fetal_brain_seg
ADD https://github.com/taigw/Demic/archive/v0.1.tar.gz /app/Demic-0.1.tar.gz
RUN cd /app && \
    tar xvf Demic-0.1.tar.gz && \
    mv Demic-0.1 /app/fetal_brain_seg/Demic

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
    # own libraries:
    nsol>=0.1.14 \
    pysitk>=0.2.19 \
    simplereg>=0.3.2
RUN pip install -e .

# prepare fetal_brain_seg with specific python library versions
COPY --from=compile-image /app/fetal_brain_seg /app/fetal_brain_seg
WORKDIR /app/fetal_brain_seg
RUN pip install \
    niftynet==0.2 \
    tensorflow==1.12.0
RUN SITEDIR=$(python -m site --user-site) && \
    mkdir -p $SITEDIR && \
    echo /app/fetal_brain_seg > $SITEDIR/Demic.pth
ENV FETAL_BRAIN_SEG=/app/fetal_brain_seg

# add Dockerfile to image
ADD Dockerfile /

WORKDIR /app
CMD bash