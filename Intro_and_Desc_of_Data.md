## INTRODUCTION AND DESCRIPTION OF DATASET

**Background -** 

Subjects in this trial wore an Apple Watch to collect their ambulatory activity patterns for a week before spending one night in a sleep lab. During that night, acceleration (in g) and heart rate (in beats per minute, bpm) were collected from the Apple Watch while they underwent polysomnography (PSG). Each type of data recorded from the Apple Watch and the labeled sleep from polysomnography is saved in a separate file, tagged with a random subject identifier.

**Understanding the Dataset -** 

The following types of data are provided:

**motion (acceleration):** Saved as txt files with the naming convention '[subject-id-number]_acceleration.txt'

Each line in this file has the format: date (in seconds since PSG start), x acceleration (in g), y acceleration, z acceleration

**heart rate (bpm):** Saved as txt files with the naming convention '[subject-id-number]_heartrate.txt'

Each line in this file has the format: date (in seconds since PSG start), heart rate (bpm)

**steps (count):** Recorded from the Apple Watch and saved in the format '[subject-id-number]_steps.txt'

Each line in this file has the format: date (in seconds since PSG start), steps (total in bin from this timestamp to next timestamp)

**labeled sleep:** Recorded from polysomnography and saved in the format '[subject-id-number]_labeled_sleep.txt'

Each line in this file has the format: date (in seconds since PSG start) stage (0-5, wake = 0, N1 = 1, N2 = 2, N3 = 3, REM = 5)

**Aims -** 

Our assignment aims to take this given time series data and clean, process and restructure it so we can use it to create and train a classification model. 

The model should be able to determine what sleep stage a subject is in based on the features seen above. 

The work that we have done is split into two separate jupyter notebooks, one for cleaning and processing, and the other for the creation and testing of our classification model.