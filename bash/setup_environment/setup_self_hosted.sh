sudo apt update
sudo apt-get --yes --force-yes install git

# création répertoire CICD:
sudo mkdir /var/data/cicd
sudo chmod 777 /var/data/cicd
cd /var/data/cicd

#anaconda (trouvé dans https://docs.anaconda.com/anaconda/install/linux/)
sudo apt-get --yes --allow install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh -b -p Anaconda3
eval "$(/home/MDaab-Admin/anaconda3/bin/conda shell.bash hook)"	# !!!marche mais dépendant du répertoire home!!! (nécessaire seulement si on n'a pas fait conda init)
eval "$(/var/data/cicd/anaconda3/bin/conda shell.bash hook)"

mkdir actions-runner
cd actions-runner

### installation du projet:
sudo apt-get --yes --allow install nvidia-cuda-toolkit
sudo apt-get install postgis

git clone https://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:x-oauth-basic@github.com/IGNF/lidar-prod-quality-control.git	# OAuth token (token généré le 2 février 2022, valide 90 jours)


### "notre" procédure (depuis le README.md):

cd lidar-prod-quality-control
# scp mdaab@HP1910P043:code/setup_env_no_gpu.sh bash/setup_environment/
source bash/setup_environment/setup_env_no_gpu.sh

### installation des fichiers en plus
mkdir dataSet
sudo chmod 777 /var/data/dataSet
scp -r mdaab@HP1910P043:/home/MDaab/Data/dataSet/ /var/data/dataSet

mkdir extra_files
# sudo chmod 777 extra_files 	# peut-être pas besoin
scp -r mdaab@HP1910P043:/home/MDaab/code/Segmentation-Validation-Model/logs/V3.0 extra_files
mkdir output_prediction
cp .env_example .env



### installation de l'action runner
/ Download the latest runner package
curl -O -L https://github.com/actions/runner/releases/download/v2.263.0/actions-runner-linux-x64-2.263.0.tar.gz

// Extract the installer
tar xzf actions-runner-linux-x64-2.263.0.tar.gz
        
// Create the runner and start the configuration experience

# cette ligne peut être récupérée dans "settings", actions, runners et "new runner"
./config.sh --url https://github.com/IGNF/lidar-prod-quality-control --token AWMVYAOSJSSOP7POSEDVNODCBIPVU




