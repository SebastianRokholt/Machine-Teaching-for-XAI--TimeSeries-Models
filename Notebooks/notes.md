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


TASK FORMULATION FOR LLM EVALUATION
There is an AI that has been trained on a large dataset of electric vehicle (EV) charging sessions. The AI is able to classify unseen charging sessions as "normal" or "abnormal" based on information such as the power transfer between the charger and battery in kW over time, the state of charge (SOC) of the battery over time, as well as other features such as temperature and charger specifications. 

You don't know anything else about the AI. 

We have provided you with 10 images (examples) of charging sessions. These examples include the power transfer in kW over time (minutes elapsed since the beginning of the charging session) and the SOC over time. These data are also given to the AI. In addition, we have provided you with a simplified version of the power curve. When given to the AI, this simplified version resulted in the exact same classification/label as for the original power curve, even though the AI never saw the original and simplified curves together as you do here. 

Based on what you can infer about the AI's behaviour from these examples and the information above, what do you think the AI would classify each example as? Provide your guesses ("normal" vs "abnormal") for all 10 examples. 

IMPORTANT: DO NOT USE INFORMATION IN THE IMAGE FILE NAME. 