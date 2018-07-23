Dear participant of AVEC 2018,

This is the data package for the Bipolar Disorder Sub-Challenge (BDS) of the 8th Audio/Visual Emotion Challenge and Workshop (AVEC 2018): "Bipolar Disorder and Cross-cultural Affect".

The AVEC 2018 BDS is a new first of its kind task in the scope of mental health analysis. 
Whereas the topic of depression analysis was featured in the previous editions of AVEC, we introduce this year for the first time in a challenge the analysis of bipolar disorder, which rank among the top-ten mental disorder for adults according to the World Health Organisation. 
The dataset used for the AVEC 2018 BDS includes audiovisual recordings of structured interviews performed by 47 Turkish speaking subjects aged 18-53 (BD corpus). 
All those subjects suffered from bipolar disorder and were recruited from a mental health service hospital where they were diagnosed by clinicians following DSM-5's inclusion criteria.

Participants of the BDS have to classify patients suffering from bipolar disorder into remission, hypo-mania, and mania, following the young mania rating scale, from audio-visual recordings of structured interviews (BD corpus). 
More information on this database can be read in its introductory paper (BD_corpus_intro.pdf). 
Please note that features and methods used in this paper to perform the classification of the bipolar disorder are not those of AVEC 2018. 
However, participants are free to use them if they want; functions and model are accessible via goo.gl/ZAm1zw. 
The BUEMODB corpus used in the paper for affect-oriented analysis of bipoplar disorder is also available upon request - please contact kaya.heysem@gmail.com for this purpose.

The official metric to evaluate the performance is the unweighted average recall: 

	UAR = 1/3 * (recall(remission) + recall(hypo-mania) + recall(mania))

The baseline system relies on SVMs (liblinear library) with one model for each modality (audio/video) and their different representations (functionals, Bag-of-Words, Deep Spectrum for audio) with either frame-based or session-based decisions. For audio data, MFCCs (frame level), eGeMAPS (turn level based on post-processed VAD obtained with LSTM RNN), Bag-of-Audio-Words (window sise of 2s, hop-size of 1s, 20 soft assignments, codebook size of 1000) and Deep Spectrum representations are used for SVMs learning. For video data, functionals of FAUs (session level) and Bag-of-Video-Words (window sise of 11s, hop-size of 1s, 20 soft assignments, codebook size of 1000) are exploited.  Standardisation is performed only on the sets of functionals (eGeMAPS for audio and FAUs for video); Bag-of-Words and Deep Spectrum representations based features are processed as they are (values of log-frequency, and activation function are naturally in the appropriate range for machine learning). For frame based features, the final decision on the session is taken by majority voting from the frame-based predictions. Performance (%UAR) on the development set is as follows (chance level is 33.33):

(DEV) MFCCs: 52.91; eGeMAPS: 55.03; BoAW: 57.67; DeepSpectrum: 52.12; FAUs: 41.80; BoVW: 53.44

Performance on the test set for the best representation for audio and video, respectively, is as follows:

(TEST) eGeMAPS: 51.85; BoVW: 25.93

Fusion of audio, visual, and audiovisual representations, which are obtained by another SVMs model learned on a posteriori probabilities estimated from the frame-based decisions, or by a sigmoid function applied to the distance tio the hyperplane for session-level features, i.e., FAUs, performs as follows:

(DEV) audio: 69.84; visual: 57.14; audiovisual: 79.37
(TEST) audio: 50.00; visual: 33.33; audiovisual: 44.44

Baseline scripts for feature extraction are available on https://github.com/AudioVisualEmotionChallenge/AVEC2018. 

The data package includes:

*** recordings ***

- recordings.zip
	zip file of speech PCM wave files and video mpeg4 files for training, development (validation) and test sets

*** features ***

- LLDs_audio_opensmile.zip
	zip file of low-level descriptors (LLDs) extracted with opensmile on each audio recording for training, development (validation) and test sets
	it includes csv files containing the following sets of audio descriptors: MFCCs (0-12, delta, delta-delta), and eGeMAPS. More information on the feature extraction toolkit opensmile can be accessed here: https://audeering.com/technology/opensmile/; details on the eGeMAPS set are accessible here: https://ieeexplore.ieee.org/abstract/document/7160715/.

- LLDs_video_openface.zip
	zip file of low-level descriptors (LLDs) extracted with openface on each video recording for training, development (validation) and test sets
	it includes a csv file containing facial landmarks, head pose, eye gaze, facial action units, and a hog file containing histogram of oriented gradient features. More information on the features can be accessed here: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format.

- baseline_features.zip
	zip file containing the sets of audiovisual features (csv files) used to run the baseline recognition system.

*** time codes ***

- VAD_turns.zip
	zip file containing turn timings of speech as csv files (start;stop); VAD was performed with the openSMILE LSTM-RNN based toolkit, and further post-processed to remove the sound separators (see below), and merge segments between short pauses.

- sound_separator.zip
	zip file containing timings between each sound separator produced during the data collection; csv files (start;stop). Those timings were obtained by template matching of the sound event and then cutting segments before and after those separators. They were originally devoted to separate the questions answered by the subjects but were used only for removing the separator sound because the number of sound separators varies according to the recordings and rarely match the number of questions.

*** labels ***

- labels_metadata.csv
	csv file containing metadata and labels for training and development sets
	metadata: SubjectID, Age, Gender, 
	labels: total YMRS (0-40), mania level (1-3)

Please note that this readme file is subject to potential changes.
Last update: 	06/06/2018: baseline feature sets and baseline system's performance
		05/10/2018: openface's LLDs
		05/08/2018: precision regarding the BD corpus paper not being AVEC 2018 baseline
		04/13/2018: creation

We wish you a very successful AVEC 2018 Challenge!
The organisers, Fabien Ringeval, Bjoern Schuller, Michel Valstar, Roddy Cowie and Maja Pantic
