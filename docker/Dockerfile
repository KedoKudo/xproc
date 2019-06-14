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
        jupyterlab \
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
    pip install tifffile docopt vtk && \
    pip install git+https://github.com/KedoKudo/tomoproc#egg=tomoproc
# config jupyter
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter lab build
RUN npm cache clean --force
