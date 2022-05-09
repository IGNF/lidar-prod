# This is a setup script for the self-hosted server 
# used for CI/CD workflows.

# Install git
sudo apt update
sudo apt-get --yes --force-yes install git

# Create CICD dir
sudo mkdir /var/data/cicd
sudo chmod -R 777 /var/data/cicd
mkdir /var/data/cicd/CICD_outputs/
mkdir /var/data/cicd/CICD_outputs/app/
mkdir /var/data/cicd/CICD_outputs/opti/
cd /var/data/cicd

# Install anaconda (https://docs.anaconda.com/anaconda/install/linux/)
sudo apt-get --yes --allow install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh -b -p Anaconda3
eval "$(/home/MDaab-Admin/anaconda3/bin/conda shell.bash hook)"	# !!!marche mais dépendant du répertoire home!!! (nécessaire seulement si on n'a pas fait conda init)
eval "$(/var/data/cicd/anaconda3/bin/conda shell.bash hook)"

# Mouting file assets for tests - you will need credentials fo both self-hosted server and mounted asset directory
sudo mount -v -t cifs -o user=cgaydon,domain=IGN,uid=24070,gid=10550 //store.ign.fr/store-lidarhd/projet-LHD/IA/Validation_Module/CICD_github_assets /var/data/cicd/CICD_github_assets/

# Install necessary libraries onces
sudo apt-get --yes --allow install nvidia-cuda-toolkit
sudo apt-get install postgis


### instal of action runner
mkdir actions-runner
cd actions-runner

# Download the latest runner package
curl -O -L https://github.com/actions/runner/releases/download/v2.263.0/actions-runner-linux-x64-2.263.0.tar.gz

# Extract the installer
tar xzf actions-runner-linux-x64-2.263.0.tar.gz
        
# Create the runner and start the configuration experience
# Copied from "settings" -> actions ->runners -> "new runner"
./config.sh --url https://github.com/IGNF/lidar-prod-quality-control --token AWMVYAOSJSSOP7POSEDVNODCBIPVU
