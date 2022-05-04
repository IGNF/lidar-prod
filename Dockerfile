FROM continuumio/anaconda3:latest

# set the IGN proxy, otherwise apt-get and other applications don't work 
# from within our self-hoster action runner
ENV http_proxy 'http://192.168.4.9:3128/'
ENV https_proxy 'http://192.168.4.9:3128/'

# set the timezone, otherwise it asks for it... and freezes
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# all the apt-get installs
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    software-properties-common \
    wget \
    git \
    postgis                     

# /lidar becomes the working directory, where the repo content 
# (where this Dockerfile lives) is copied.
WORKDIR /lidar
COPY . .

# install the python packages via anaconda
RUN conda env create -f bash/setup_environment/requirements.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "lidar_prod", "/bin/bash", "-c"]

# test if pdal is installed (a tricky library!)
RUN echo "Make sure pdal is installed:"
RUN python -c "import pdal"

# the entrypoint garanties that all command will be runned in the conda environment
ENTRYPOINT ["conda", \
    "run", \
    "-n", \
    "lidar_prod"]

# Example command to run the application from within the image
CMD  ["python", \
    "lidar_prod/run.py", \
    "print_config=true", \
    "paths.src_las=your_las.las", \
    "paths.output_dir=./path/to/outputs/"]
