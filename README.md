# face_recognition
This is a private repository for Smart City project in the State University of Telecoms of Saint-Petersburg. For more information, see the target.md file

## Installation
* Download this repository
* Create a virtual environment with python virtualenv by the following command:
    
```
    virtualenv --python=/usr/bin/python3 venv
```
* Activate the environment with:
 
For Bash Users:
```
    source venv/bin/activate
```

For Fish Users:
 ```
    source venv/bin/activate.fish
 ```

## Usage

To use the video_recognition script, use the following command:
```
python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
```


## Collaborators
* Julien Aldon
* Julie Wurtz
* Valetin Collomb
* Enzo Nicoletti