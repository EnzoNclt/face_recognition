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

To detect on a camera the age and gender of one or several persons, you will need to use the detect_faces_video script, use the following command:
```
python detect_faces_video.py [--confidence conf(default=0.5)]
```

The face recognition is also availible on a single picture, using this command:
```
python detect_faces.py --image path_to_image
```

You can also try the gender age detection on a single image with the script gad using the following script
```
python gad.py --image path_to_image
```

## Collaborators
* Julien Aldon
* Julie Wurtz
* Valetin Collomb
* Enzo Nicoletti