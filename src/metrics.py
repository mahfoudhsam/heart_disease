import numpy as np

def accuracy(y_true,y_pred):
    if len(y_pred)==0:
        return 0
    else:
        return np.sum((y_true==y_pred))/len(y_true)
def precision(y_true,y_pred):
    tp=np.sum((y_true==1) & (y_pred==1))
    fp=np.sum((y_true==-1) & (y_pred==1))
    if (tp+fp)==0:
        return 0
    else:
        return tp/(tp+fp)
def recall(y_true,y_pred):
    tp=np.sum((y_true==1) & (y_pred==1))
    fn=np.sum((y_true==1) & (y_pred==-1))
    if (tp+fn)==0:
        return 0
    else:
        return tp/(tp+fn)
def F1_score(y_true,y_pred):
    p=precision(y_true,y_pred)
    r=recall(y_true,y_pred)
    if (p+r)==0:
        return 0
    else:
        return (2*p*r)/(p+r)
def Confusion_Matrix(y_true,y_pred):
    tp=np.sum((y_true==1) & (y_pred==1))
    tn=np.sum((y_true==-1) & (y_pred==-1))
    fp=np.sum((y_true==-1) & (y_pred==1))
    fn=np.sum((y_true==1) & (y_pred==-1))
    return np.array([[tn,fp],[fn,tp]])

    