<p align="center">
  <img src="readme_images/shhhh-cat-sigma.png" alt="shhh" width="420" />
</p>

Project: Personal Silent Speech Recognition System (SSR) — DIY AlterEgo
Mission: Build a working silent speech recognition system trained on a single user (me), reaching AlterEgo-level quality (target ~92% accuracy on a personal vocabulary). This is a research/personal project — MVP is a system that works on me specifically, no generalization required. End goal is eventually a silent universal translation system and invisible AI assistant interface.
Hardware:

OpenBCI V3 compatible 8-channel EEG/EMG board (Arduino-based)
OpenBCI Gold Cup Electrodes x10
ELEGOO DuPont jumper wires (for connections/adapters)
Nexcare medical paper tape (electrode securing)
Alcohol prep pads (skin prep)
PSIER Bone Conduction Headphones (Bluetooth, for audio feedback closing the loop)
Technical approach: EMG-based subvocalization detection, same fundamental method as MIT AlterEgo (2018 paper). Electrodes placed on jaw/face muscles — mentalis (chin), masseter (jaw hinge), with mastoid bone reference electrode for noise rejection. NOT EEG/brainwave reading, purely surface EMG of speech muscles.
Planned ML architecture: Two-model cascade:

Model 1 trained on overt mouthing (stronger signal, easier, bootstrapping model)
Model 2 trained on pure subvocalization (zero visible movement), using Model 1 predictions as soft labels for self-supervised training At inference: ensemble both models, use signal amplitude to switch between them automatically
Software stack:

OpenBCI GUI for initial signal verification and data streaming
Python for data pipeline (bandpass filter 20-500Hz, epoch extraction, feature engineering)
ML framework TBD (PyTorch likely)
Compute: M2 Pro Mac mini + Ubuntu workstation + cloud/agent fleet available
Current status: All hardware just arrived, have not done first session yet. Complete beginner on the hardware side but fast learner, strong AI/ML background, has access to AGI-class coding and research agent fleets for the software side.
Immediate next steps:

Install OpenBCI GUI, verify board streams data on all 8 channels
First electrode placement session — verify clean EMG burst on jaw clench
Begin labeled data collection on 10-word vocabulary, 200+ reps each
Build Python recording and preprocessing pipeline
Train Model 1 on mouthing data
Iterate toward subvocalization Model 2 User context: Fast learner, understands AI concepts deeply, needs hardware guidance kept practical and step-by-step for now. Prefers direct technical communication, no hand-holding on the software/ML side. Goal is to move fast and iterate. Document everything for potential future publication — the dual-model self-supervised bootstrapping approach is potentially novel enough to publish if it works.