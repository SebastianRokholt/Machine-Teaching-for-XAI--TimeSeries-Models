# MT4XAI study design
## Research question and hypotheses
RQ: “How can techniques from machine teaching for XAI be applied to
time series classifiers in order to generate simple, understandable
and faithful explanations of model decisions in a real world industry
setting?”

The user study will be designed to answer the following hypotheses:
• H1: Participants who receive ORS-based simplifications will classify abnormal ses-
sions more accurately than those who see raw time series.
• H2: Teaching sets ordered by a simplicity prior (curriculum learning) will improve
human learners’ ability to simulate the decisions of a Time Series Classifier on EV
charging data, compared to unordered teaching sets.


## Test groups for user study
- Group A: Raw series + simplification (superimposed) and curriculum learning w/simplicity prior. 
- Group B: Raw series + simplificiation (superimposed). Unordered
- Group C: Only raw series for teaching. Random sampling

## Phases: 
Phase 1: Pre-teaching exam
    Items: Exam set 1 or 2. 
    Group A: overlay
    Group B: overlay
    Group C: raw only
    Logging: {participant_id, exam_set_id, item_id, AI_class, k, margin, modality_shown, learner_guess, response_time_ms}.

Phase 2: Teaching. Alternate between classes. Show label (what the AI predicted)
    Group A: overlay + curriculum (k ascending; tie-break by Macro-RMSE margin relative to the anomaly threshold).
    Group B: overlay + unordered.
    Group C: raw-only teaching, unordered
    Logging: {participant_id, item_id, AI_class, k, margin, modality_shown, time_spent_on_item}

Phase 3: Post-teaching exam
    Items: Opposite exam set as used in pre-teaching exam (1 or 2)
    Group A: overlay
    Group B: overlay
    Group C: raw only
    Logging: {participant_id, exam_set_id, item_id, AI_class, k, margin, modality_shown, learner_guess, response_time_ms}.

## Exam sets
Two disjoint sets that are matched in terms of filtering and balance across AI class and simplicity k.  
Which form is provided first and last is determined by random sampling (50% chance of either) per trial. 
The exam set should consist of 20 examples (10 normal, 10 anomalous charging sessions).
For each example, the user/learner is asked: "What do you think the AI classified this charging session as?"
The answer is provided as a 50/50 multiple choice: "normal" vs "abnormal".

## Teaching set (S)
### S for group A: 
Examples are raw series w/simplification (overlay). Order the teaching set ascending by simplicity (k) and then descending by margin for tie breaks. Margin is a secondary difficulty metric: threshold − error for normal; error − threshold for abnormal, so higher margin value means “easier.”
### S for group B: 
Examples are raw series w/simplification (overlay). Examples are selected with random sampling (unordered). 
### S for group C: 
Examples are raw series, and are selected with random sampling (unordered). 
### Size (cardinality) of S: 
50 examples. All three teaching sets consist of the same charging session IDs (raw series). 
### Selection technique: 
    1. Classify all charging sessions in the test set (12,183 sessions). Store class labels. 
    2. Calculate simplifications for all charging sessions. Store simplifications along with their simplicity (k), robustness, fidelity and margin (threshold - error for nomal; error - threshold for abnormal) metrics. 
    3. Order all sessions by k (ascending) and margin (descending). 
    4. 