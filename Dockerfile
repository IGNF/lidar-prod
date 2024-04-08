FROM mambaorg/micromamba:latest

# all the apt-get installs
USER root

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    software-properties-common \
    postgis

# Only copy necessary files to set up the environment, in order
# to use docker caching if requirements files were not updated.
# Dir needs to be "/tmp" for micromamba to find the pip requirements...
WORKDIR /tmp
COPY ./setup_env/ .

# install the python packages via anaconda
RUN micromamba create --file /tmp/requirements.yml

# Sets the environment name (since it is not named "base")
# This ensures that env is activated when using "docker run ..."
ENV ENV_NAME lidar_prod
# Make RUN commands here use the environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
# List packages and their version
RUN micromamba list
# test if pdal is installed (a tricky library!)
RUN echo "Make sure pdal is installed:"
RUN python -c "import pdal"

# /lidar becomes the working directory, where the repo content
# (the context of this Dockerfile) is copied.
WORKDIR /lidar
COPY . .

# Example command to run the application from within the image
CMD  ["python", \
    "lidar_prod/run.py", \
    "print_config=true", \
    "paths.src_las=your_las.las", \
    "paths.output_dir=./path/to/outputs/"]
