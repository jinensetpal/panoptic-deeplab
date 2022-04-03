git clone https://github.com/jinensetpal/panoptic-reproducibility

apt update
apt install python3.10 python3.10-distutils python3.10-dev cuda-toolkit-10-2
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.10 get-pip.py
pip3.10 install -r requirements.txt
pip3.10 install dvc mlflow numpy==1.21.5 ipykernel

tar -xvf cudnn.tar.gz
cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda:/usr/local/cuda/include:/usr/local/cuda/lib64/
ldconfig

dvc remote modify origin --local auth basic
dvc remote modify origin --local user $DAGSHUB_UNAME
dvc remote modify origin --local password $DAGHUB_TOKEN
dvc pull -r origin

MLFLOW_TRACKING_URI=https://dagshub.com/jinensetpal/panoptic-reproducibility.mlflow \
MLFLOW_TRACKING_USERNAME=$DAGSHUB_USER_NAME \
MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN \
python3.10 -m src.models.deeplab

dvc add models 
dvc commit
git commit -a -m "added trained model"
git push origin main 
