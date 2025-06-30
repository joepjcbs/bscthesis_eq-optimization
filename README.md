# BSc Thesis: Subject-specific Optimization of Equalizer Band Gains in Auditory Brain-Computer Interfaces

This repository contains the newly introduced Python modules from my Bachelor's thesis. While initially designed to function within the refactored Python version of the aphasia rehabilitation codebase used by Simon Kojima et al. (2024), the modules should function with any auditory stimuli presented through PyAudio. The current classification pipeline uses XDF files as input, but could easily be edited to incorporate EEG files. 

#### Overview of modules:
- Session Manager: keeps track of participant names, folders, logging, etc.
- EQ: creates and applies filter bank.
- Analysis: classification of EEG data, returns single-trial AUC scores.
- Optimization: takes AUC scores from analysis.py and predicts the optimal configuration using Gaussian process regression.


#### References
Kojima, S., Kortenbach, B. E., Aalberts, C., Miloševska, S., de Wit, K., Zheng, R., Kanoh, S., Musso, M., & Tangermann, M. (2024). Inﬂuence of pitch modulation on event-related potentials elicited by Dutch word stimuli in a brain-computer interface language rehabilitation task.
