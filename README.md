# Machine learning in the Medical Field
## Team and Responsibilities
Brendon Mustachio - Module designer, developer, and researcher

Selvin White - Module designer, developer, and researcher

Our team will collaborate on building and evaluating a machine learning system that recognizes surgical gestures from recorded Da Vinci motion data and identify diseases and ailments. Responsibilities will include data preprocessing, model development, performance testing, results analysis, and presentation/report preparation.

## Feedback Received and Responses
- Identify specific gestures during procedures.
- Provide an experimental setup + how it will be demonstrated.

## Problem and Motivation
Surgeons learning robot-assisted procedures need detailed feedback on their tool movements and surgical technique. Much of this feedback is given after a training session, which can delay their improvement. Our project investigates whether machine learning can recognize gestures quickly enough to provide useful real-time feedback during simulated Da Vinci training.
This is a machine learning problem because the model must balance gesture recognition accuracy with inference latency, memory use, and model size. A model that is accurate but too slow would not be useful for real-time feedback.

## Research Questions and Hypotheses
| Research Questions | Hypotheses | 
| --- | --- |
|Can machine learning accurately recognize surgical gestures from recorded data? | Temporal machine learning models will recognize common surgical gestures with higher macro F1-scores than a simple baseline model.|
|Can a lightweight model provide feedback quickly enough for real-time training? | A smaller temporal model will have lower average and p95 latency than a larger model while maintaining useful recognition accuracy. |
| Does the amount of recent motion data affect recognition quality? | Using a longer input window will improve gesture recognition quality up to a point, but will increase latency and memory use.| 
| Are some surgical gestures harder for machine learning to recognize than others? | Similar gestures with overlapping tool movements will be confused more often than clearly different gestures.|

## Related Work
Machine learning is surgery is being explored by medical professionals. Dr. Francis Creighton, a surgeon scientist at Johns Hopkins Medicine and is leading in research to develop cooperative robots that can help surgeons avoid critical structures. These robots could provide real-time feedback, reduce tremor, and help surgeons avoid dangerous structures. His work supports the idea that AI should assist surgeons with information and feedback rather than replace their medical judgement.
(Link:) https://www.hopkinsmedicine.org/news/articles/2024/02/using-ai-to-enhance-lateral-skull-base-surgery

The Da Vinci platform is an appropriate example for this project because it is used by trained physicians to control endoscopes and surgical instruments during minimally invasive procedures. Intuitive Surgical also describes Da Vinci 5 as a platform designed for surgical data analytics, objective skill assessment, and future AI and machine learning capabilities. Our project builds on this direction by evaluating whether gesture recognition models can produce useful low latency feedback from recorded motion data.
(Link:) https://www.intuitive.com/en-gb/products-and-services/da-vinci/5-teaser

## Proposed System or Approach
We will create a prototype that receives recorded Da Vinci robot motion data, preprocesses the data into sequences of tool movements, and uses machine learning models to classify the current surgical gesture. The prototype will output the predicted gesture and confidence score as simulated real-time feedback for a trainee. We will compare a simple baseline model with temporal machine learning models, such as an LSTM/GRU and a temporal convolutional network as well as use recorded training data.

## Evaluation Plan
- Data: Recorded Da Vinci surgical training motion data for suturing, knot tying, and needle passing.
- Baselines: Logistic regression or random forest.

## Expected Deliverables
- Cleaned and preprocessed Da Vinci motion data pipeline.
- Baseline and temporal gesture recognition models.
- Real time inference simulation with predicted gesture and confidence.
- Evaluation results, charts, tables, and confusion matrix.
- Reproducible source code along with step by step instructions.
- A final report and presentation.

## Timeline and Milestones
| Time | Milestone |
| --- | --- |
| Weeks 1-2 | Obtain data, review related work, and set up an environment |
| Weeks 3-4 | Preprocess motion data and create a baseline model |
| Weeks 5-6 | Implement temporal machine learning models |
| Weeks 7-8 | Measure accuracy, latency, memory use, and throughput |
| Weeks 9-10 | Analyze results, create charts, and introduce the final prototype |
| Final Week | Complete report, presentation, and reproducibility materials |

## Risks and Mitigations
| Risk | Mitigation |
| --- | --- |
| The dataset is difficult to process | Begin with one task, such as suturing, before adding other tasks to process |
| Limited data causes over-fitting | Use separate surgeons when obtaining training data and test new data where possible |
| Deep learning model is too slow | Test smaller models, shorten input windows, and lower precision inference |
| Gesture labels are uneven | Report macro F1-score, not on accuracy alone |
| Results are not clinically validated | Clearly state that the project is for training feedback |
| Insufficient storage or memory for the full dataset | Use only motion data files needed for the selected tasks, compress or store data efficiently, and process data in smaller batches instead of loading everything at once. | 


## Reproducibility Plan
- Dataset download and preprocessing instruments
- Saved model configurations and package versions
- Automated training and evaluation scripts
- Fixed train/test splits 

## References​


