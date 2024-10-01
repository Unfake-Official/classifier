<h1 align="center" style="font-weight: bold;">🤖 Classifier</h1>

<div align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="python"/>
  <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" alt="tensorflow"/>
  <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white" alt="keras"/>
</div>

<p align="center">
 <a href="#about">About</a> • 
 <a href="#started">Getting Started</a> • 
  <a href="#training">Training</a> • 
  <a href="#inference">Inference</a> •
 <a href="#contribute">Contribute</a>
</p>

<h2 id="started">📌 About the project</h2>
<p>
The growing evolution of artificial intelligence (A.I.) tools is making them increasingly efficient and globally accessible. However, some of these technologies can be harmful if used maliciously, and that includes deepfakes. Deepfakes are a type of synthetic media that generates realistic content and has the potential to clone an individual's identity, using it to spread fake news, damage their reputation and promote fraud and security breaches. Thus, there is a need for ways to verify whether a piece of media is real or artificially synthesized. However, even though there are technologies that meet this need, the detection of audio deepfakes is still a challenge, considering that it is not as effective when it comes to speech in Portuguese and has questionable effectiveness in audio with the presence of noise. In this sense, Unfake aims to develop an A.I. model capable of identifying whether an audio contains human or synthetic speech. In this way, we hope to make it possible for lay users to identify deepfakes in a robust and effective way, contributing to a safer and more reliable digital environment, as well as encouraging future research in the area using the data obtained in the project.</p>
<br>

<h2 id="started">🚀 Getting started</h2>

<h3>Prerequisites</h3>

Here is a list of all prerequisites necessary for running the project locally:

- [Python](https://www.python.org)

<h3>Cloning</h3>

```bash
git clone https://github.com/Unfake-Official/classifier.git
```

<h3>Starting</h3>

Firstly, create a virtual environment and activate it: 
```bash
python -m venv .venv
.venv/Scripts/activate
```

Next, install all dependencies: 
```bash
pip install -r requirements.txt
```

<h2 id="training">Training</h2>

If you want to use our preprocessed dataset for training, download it from [portufake repository](https://huggingface.co/datasets/unfake/portufake) and unzip it.

Now, for training the model with your own data, you need to create a folder containing two subdirectories named:
- real: contains real speaker recording spectrograms
- fake: contains audio deepfake spectrograms

Then, go to ```cnn/train.py``` and change:
1. ```EPOCHS, BATCH_SIZE and VALIDATION_SPLIT``` to the values you want to use as hyperparameters.
2. ```IMG_SIZE``` to the target image size in the format ```(WIDTH, HEIGHT)```.
3. ```CHECKPOINT_PATH``` to the model's path, if you want to train from a checkpoint.
4. ```METRICS_PATH``` to the path of the image where the accuracy and loss charts will be plotted.
5. ```CSV_PATH``` to the path of the csv file where the accuracy and loss charts will be saved.
6. ```DATASET_PATH``` to the path of the dataset containing the real and fake folders as mentioned above.

Next, run the file: 
```bash
python cnn/train.py
```

The model will be saved after each epoch. You can track the training process through the terminal, with messages and progress bars to help the process.

<h2 id="inference">Inference</h2>

If you want to use our pretrained model for inference, download it from [unfake repository](https://huggingface.co/unfake/unfake), 
unzip it and place it in the following directory: ```cnn/checkpoints```

Now, if you want to use your own model, be aware that the code expects a ```tf.keras``` model saved with ```model.export()```.

Then, go to: ```cnn/inference.py``` and change: 

1. ```CHECKPOINT_PATH``` to the model's path (example: ```cnn/checkpoints/unfake```)
2. ```AUDIO_PATH``` to the audios's path you want to classify as a real recording or a deepfake (all common audio formats are accepted, such as .wav, .mp3, .ogg and .flac).

Next, run the file: 
```bash
python cnn/inference.py
```

The response will be printed as it follows:
```Your audio is probably <real/false> with <percentage>% confidence.```

<h2 id="contribute">📫 Contribute</h2>

If you want to somehow contribute to this project, start by creating a branch named as follow. Then, make your changes and follow commit patterns. Finally, open an pull request. 

1. `git clone https://github.com/Unfake-Official/classifier.git`
2. `git checkout -b feature/NAME`
3. Follow commit patterns
4. Open a Pull Request explaining the problem solved or feature made, if exists, append screenshot of visual modifications and wait for the review!

<h3>Documentations that might help</h3>

[📝 How to create a Pull Request](https://www.atlassian.com/br/git/tutorials/making-a-pull-request)

[💾 Commit pattern](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716)
