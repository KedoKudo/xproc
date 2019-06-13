FROM  continuumio/miniconda3:latest
LABEL version="0.0.1" \
      maintainer="kedokudo <chenzhang8722@gmail.com>" \
      lastupdate="2019-06-13"
USER  root
WORKDIR /home

# Install necessary libraries from offical repo
RUN apt update  -y && apt upgrade -y && \
    apt install -y  build-essential git ffmpeg
# config conda
RUN conda update conda -y -q && \
    conda update -y -q --all && \
    conda install --yes \
        jupyter \
        numpy \
        scipy \
        pandas \
        scikit-learn \
        matplotlib \
        seaborn              && \
    conda install -y -c \
        conda-forge tomopy   && \
    conda clean --all -f -y
# config pip
RUN pip install --upgrade pip && \
    pip install tifffile docopt vtk
# config jupyter
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN npm cache clean --force
