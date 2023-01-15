# ID2223 - Lab 2 Whisper

The goal of the lab was to fine-tune a pre-trained transformer model called Whisper in order to perform speech-to-text for the Swedish language. The lab consists of three parts, namely:

1. A feature pipeline that downloads data from the common voice dataset (swedish subset) and extracts labels and features.
2. A training pipeline that downloads the data from Google Drive, in this case, and trains on that data. The model is then evaluated with the evaluation metric WER (word error rate). The checkpoints were saved every 500 steps on Google Drive instead of every 1000 as in the original code (https://github.com/ID2223KTH/id2223kth.github.io/tree/master/assignments/lab2)
3. A Hugging Face UI was then lastly created to allow users to speak directly into the microphone and transcribe the speech. Additionally, it is possible to input a YouTube video to the model which is then transcribed as well. The speech-to-text also comes with a translator which translates the spoken audio into of of five languages: English, Spanish, Dutch, French and Italian.

# Hugging Face UI

The Hugging Face space can be found here: https://huggingface.co/spaces/freeja/lab2-whisper

# Model-centric improvements

- Using a larger model, such as whisper-medium or whisper-large, although this requires more GPU resources on Google colab and therefore an increased training time.
- Fine-tuning the hyperparameters, through grid-search for example, such as learning_rate (too small - slow training, too fast - not optimal learning) or dropout. Increase num_train_epochs to hopefully increase performance (will require further resources)
- Selecting a different loss function in hopes of a better performance.

# Data-centric improvements

- Increase the training dataset with additional data from a database, for example the NST Swedish ASR database (https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-56/)
- Adding additional sources such as youtube videos, user recorded speech or other databases.
- Up-scoring training data to allow the model to train on more uncommon phrases.
