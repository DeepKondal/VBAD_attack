# VBAD - Pipeline

This is a custom pipeline for testing a Black-box Adversarial Attacks on Video Recognition Models. The Black-Box attacking method used is VBAD from https://github.com/Jack-lx-jiang/VBAD a State-of-the-Art method in this domain. 

## 1. Setting up your environment

### Clone repository and enter the directory
```
git clone https://github.com/Rat-fi/VBAD-Pipeline.git
cd VBAD-Pipeline
```

### Create python virtual environment
This may vary depending on your system
```
python3 -m venv venv
source venv/bin/activate
```
Download the required librairies
```
pip install -r requirements.txt 
```


## 2. Preparing to run the pipeline (Optional) 
This step is for the users who would like to use other videos than the ones provided in the ```videos``` folder.

### Download Kinetics-400 videos dataset for testing
If you do not have your own videos, you can download some from the Kinetics-400 dataset, they are the ones used to train the I3D model. Find more information on how to do that here: https://github.com/cvdfoundation/kinetics-dataset.git

### Convert mp4 Video to npy
This will convert your mp4 video to a npy file and save it to the ```/videos``` folder, keeping the same name.
```
python3 convert_mp4_to_npy.py --video <path_to_video>
```

### Predict the class of your video
This is a necessary step if you do not know the I3D/Kinetics classification of your video, since it will be use for evaluating a successful attack using VBAD.
The path to pass is the one of the npy file you created in the previous step
```
python predict_class.py --video <videos/path_to_npy_file>
```

## 3. Running the pipeline

### Run untargeted mode
This mode returns the number of queries it took for the adversarial attacking method to lead the I3D model into missclassifying the video. A successful classification is considered to be any other classification than the original label of the video. 
```
python main.py --gpus 0 --video <videos/path_to_npy_file> --label <true_label> --adv-save-path videos/output_video.npy --untargeted --sigma 1e-3
```
```--gpus``` the number of the gpus you are using   
```--video``` path to npy file (video converted above)   
```--label``` the I3D classification of the video (needed to compare to the classification after each attack attemp)   
```--adv-save-path``` path of the saved video after the attack   
```--untargeted``` mode of the attack   
```--sigma``` amount of noise added during the attack process (1e-3 is moderate, 1e-6 is small, 1e-1 is high)

### Exemple Run
This run is attacking the video ```push_up_260.npy``` with a true label of ```260``` which is equivalent to ```push-up``` classification
```
python main.py --gpus 0 --video videos/push_up_260.npy --label 260 --adv-save-path videos/output_video.npy --untargeted --sigma 1e-3
```

### Notes:
Depending on the class/label attacked, the method will take more time, since the difficulty is not the same for each one.   
Exemples:
- ```193.npy``` will be quick
- ```push_up_260.npy``` will be longer

**This is also dependant on your GPU**
python main.py --gpus 0 --video videos/push_up_260.npy --label push_up --adv-save-path videos/output_video.npy --untargeted --sigma 1e-3