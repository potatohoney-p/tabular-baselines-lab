"""Evaluate centroid classification on a stratified held-out partition."""
import json
import random
import sys
from core import matrix,fit_centroids,classify
from features import fit_scaler,transform

def stratified_split(labels,fraction=0.25,seed=7):
    if not 0<fraction<1: raise ValueError('Fraction must be between zero and one')
    groups={}
    for index,label in enumerate(labels): groups.setdefault(str(label),[]).append(index)
    if len(groups)<2 or any(len(group)<2 for group in groups.values()):
        raise ValueError('Each of at least two classes needs two observations')
    rng=random.Random(seed); train=[]; test=[]
    for label in sorted(groups):
        group=groups[label][:]; rng.shuffle(group)
        count=max(1,min(len(group)-1,int(len(group)*fraction)))
        test.extend(group[:count]); train.extend(group[count:])
    return sorted(train),sorted(test)

def evaluate(rows,labels,fraction=0.25,seed=7):
    rows=matrix(rows)
    if len(rows)!=len(labels): raise ValueError('One label required per row')
    labels=[str(label) for label in labels]
    train,test=stratified_split(labels,fraction,seed)
    training=[rows[i] for i in train]; scaler=fit_scaler(training)
    model=fit_centroids(transform(training,scaler),[labels[i] for i in train])
    predictions=[classify(model,row) for row in transform([rows[i] for i in test],scaler)]
    confusion={label:{other:0 for other in sorted(model)} for label in sorted(model)}
    for index,prediction in zip(test,predictions): confusion[labels[index]][prediction]+=1
    correct=sum(labels[i]==p for i,p in zip(test,predictions))
    return {'accuracy':correct/len(test),'test_size':len(test),'confusion':confusion,
            'train_indices':train,'test_indices':test}

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('Usage: python evaluate.py examples/input.json')
    with open(sys.argv[1]) as stream: data=json.load(stream)
    print(json.dumps(evaluate(data['rows'],data['labels']),sort_keys=True))
