# DeepFakeDetector

## Autori projekta

- Milan Pavlov SW35/2017
- Sara Miketek SW62/2017
- Vladimir Rodušek SW23/2017

## Opis

Projekat sadrži module za formiranje i detekciju deepfake video zapisa.

Naredne notebook datoteke predstavljaju korake izvršavanja formiranja deepfake video zapisa:
  
  - [mtcnn_extract_faces.ipynb](https://github.com/roki021/DeepFakeDetector/blob/master/mtcnn_extract_faces.ipynb)
    - Ekstraktovanje lica iz video zapisa koristeći MTCNN.
  - [gan_training.ipynb](https://github.com/roki021/DeepFakeDetector/blob/master/gan_training.ipynb)
    - Treniranje GAN mreže nad slikama lica napravljenih iz prethodnog koraka ili nad već postojećim podacima
  - [video_creation.ipynb](https://github.com/roki021/DeepFakeDetector/blob/master/video_creation.ipynb)
    - Formiranje deepfake video zapisa

Naredne notebook datoteke predstavljaju deo vezan detekcije deepfake video zapisa:

  - [train_meso4.ipynb](https://github.com/roki021/DeepFakeDetector/blob/master/train_meso4.ipynb)
    - Treniranje Meso4 mreže nad skupom podataka
  - [check_video.ipynb](https://github.com/roki021/DeepFakeDetector/blob/master/check_video.ipynb)
    - Provera da li je video zapis deepfake ili realan

### Skup podataka za proveru rada formiranja deepfake video zapisa

Potrebne modele za GAN mrežu kao i 2 video zapisa na osnovu kojih je dobijen skup slika nad kojima je mreža trenirana, može se preuzeti sa sledećeg link-a: 

[Skup podakata](https://www.kaggle.com/mikecp/faceswap)
