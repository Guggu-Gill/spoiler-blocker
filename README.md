# Spoiler Blocker using Universal Sentence Encoder(USE) Embeddings


1. Architecture of USE

![Universal-sentence-encoder-multi-task-and-deep-averaging-network-48](https://github.com/Guggu-Gill/spoiler-blocker/assets/128667568/e953dc3f-411e-45d9-b455-fa0fafec02a2)

https://arxiv.org/abs/1803.11175

Why USE?
- Firstly its simplier to use it automatically computes word embeddings & bigrams and outputs sentence embeddings
- Its better than average of word embeddings cause bigrams are also into consideration while producing embeddings
- It gives unified 512 dim vector


2. Contrastive Learning Deep Model
- reviews and plots were embedded using USE embedings
- It inputs both reviews and plot to measure similarity
- there is not very very major imbalance problem it can be handled using class weights in keras(balancing loss function).
- 40 epochs was warmup period after that early stopping comes into picture to check overfitting in model.
- multiple batches were run [128, 256, 512] to find best metrics.

![7063416e-0793-4c64-9f76-f64d8b18dd95](https://github.com/Guggu-Gill/spoiler-blocker/assets/128667568/3c7fd5d6-178a-4db9-9647-0168ae22aa9e)


![49b963ef-4d8c-4e42-8864-29c09fa225c4](https://github.com/Guggu-Gill/spoiler-blocker/assets/128667568/53c9a752-4109-40f6-8565-8ae3e2b13e3c)

3. Results
 

Classification Report:
               precision    recall  f1-score   support

           0       0.84      0.80      0.82     84609
           1       0.50      0.58      0.54     30173
    accuracy                           0.74    114782
   macro avg       0.67      0.69      0.68    114782
weighted avg       0.75      0.74      0.74    114782





![af3b29f1-c34c-4baa-aa1b-26c678971496](https://github.com/Guggu-Gill/spoiler-blocker/assets/128667568/37e817e6-06a7-411d-b1a5-4ca16e31bf5d)
![7a7a5959-dc53-44ef-aba0-bc1cf6885005](https://github.com/Guggu-Gill/spoiler-blocker/assets/128667568/2da1e9e6-c2c1-4663-8cd7-c700d6fa5ee7)


