#M3L

Code for the paper **Multimodal Meta-Learning through Meta-Learned Task Representations**
accepted to Neural Computing and Applications journal (2024).

we propose a meta-learning framework that can handle multimodal task distributions by 
conditioning the model on the current task, resulting in a faster adaptation.
Our proposed method learns to encode each task and generate task embeddings that modulate 
the model’s activations. The resulting mod ulated model become specialized for the current 
task and leads to more effective adaptation.

# Usage
To run the code use **regression_train.py** file for regression experiments and
**classification_train.py** file for classification experiments. 
Results in the paper are obtained using the "Conditioning with sigmoid" method, but
other conditioning approaches can be find in the training files.

For testing use **regression_eval.py** and **classification_eval.py**.