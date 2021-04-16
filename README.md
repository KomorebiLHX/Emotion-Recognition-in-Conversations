# Emotion Recognition in Conversations
The project is designed to recognize the emotion in conversations(IEMOCAP Dataset), and it is also my graduation thesis.
<br />Use DialogueRNN, DialogueGCN, BERT_BASE and RGAT.

## Performance
|  Model  |  Experimental F1 Score  |  Reported F1 Score in paper  |
|  :----:  | :----:  | :----:  |
|  DialogueRNN  |  62.15  |62.75  |
|DialogueGCN|64.08|64.18|
|BERT_BASE|53.72|53.31|
|RGAT(with position encodings)|62.18|65.22|

Since the optimal parameters are not given in the paper and the experiment has a certain degree of randomness,
there is a gap between the reproduced model and the score reported in the paper.

## Paper
[DialogueRNN](https://arxiv.org/pdf/1811.00405.pdf) 
<br />[DialogueGCN](https://arxiv.org/pdf/1908.11540.pdf)
<br />[BERT_BASE](https://arxiv.org/pdf/1810.04805.pdf)
<br />[RGAT](https://www.aclweb.org/anthology/2020.emnlp-main.597.pdf)

## Dataset
IEMOCAP is an audiovisual database consisting of recordings of ten speakers in dyadic conversations. The utterances are annotated with one of six emotional labels: happy, sad, neutral, angry, excited, or frustrated.
<br />If you want to know more about IEMOCAP, please refer to [IEMOCAP](https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Busso_2008_5.pdf).
<br />If you want to run the programs of DialogueRNN and DialogueGCN, please refer to 
[DialogueGCN/IEMOCAP_features](https://github.com/declare-lab/conv-emotion/tree/master/DialogueGCN/IEMOCAP_features) 
and
[DialogueRNN/DialogueRNN_features.zip](https://github.com/declare-lab/conv-emotion/blob/master/DialogueRNN/DialogueRNN_features.zip).

## RGAT
RGAT use position encodings. If you want to use different type of PE, you can change the variable called encoding in main.py.
Relation Position Encodings is proved to be the best position encoding.

## Environment
You need to install these packages:
<br />Pytorch, Transformers, PyTorch Geometric

## Run
To run the program, you can download the folder of the model you want to test and the data file, then type in the terminal
```
python main.py
```

## Statistic
I also analyzed the IEMOCAP dataset, see the folder called statistic for details.(Please download IEMOCAP_features_bert.pkl)
