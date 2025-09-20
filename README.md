# Hierarchical Modeling of Human Language Processing with Large Language Model for Multimodal Depression Detection

#### by- Jihun Lee


+ ***Ms Student, Dept. of IT Fusion Technology***  
+ ***Chosun University, Gwangju, Rep. of Korea***

## Task- At A Glance:

1. __Task__: Depression severity detection (PHQ-8 regression)
2. __Input__: Audio, Text
3. __Output__:  4 Class (Stress and other)
4. __Database__: 2, (1) _DAIC-WOZ_, (2) _E-DAIC_

## Publication
TBD

#### Abstract
Depression is a prevalent mental disorder that impairs emotional and physical functioning, substantially reducing quality of life. Early diagnosis and accurate prediction are crucial to mitigate long-term impact. Previous studies have leveraged audio and text but typically treats them in isolation, overlooking the cognitive-interpretation-expression processes underlying conversation. We present the Cognition Interpretation-Expression Depression Network (CIEDep-Net), a multimodal framework that functionally models this three-stage structure. Each stage analyzes inputs with features tailored to its role, and a strategy combining score-conditioned fusion and cross-attention captures inter-stage interactions. In the Interpretation stage, a chain-of-thought large language model (LLM) infers depression scores and generates summaries from transcripts. These outputs correlate with ground-truth PHQ-8 score (Pearson’s r = 0.72, p < 0.01), supporting predictive validity. Empirically, CIEDep-Net achieved MAE 2.08, RMSE 3.1 on DAIC-WOZ, and MAE 2.4, RMSE 3.7 on E-DAIC, a 15.8% MAE reduction over the strongest prior multimodal baseline. Ablations confirm that removing any stage degrades performance, underscoring complementary contributions of cognition, interpretation, and expression. By embedding human language processing mechanisms into multimodal learning, CIEDep-Net delivers reliable, consistent depression-severity prediction across datasets. The approach suggests a path toward clinically meaningful, scalable assessment by integrating linguistic and paralinguistic cues while leveraging text-based intermediate representations.

## Requirements
Use the "requirements.txt" file.
