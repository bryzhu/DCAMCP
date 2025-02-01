from io import SEEK_CUR
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from model_kan_attention_raw_fingerprint_dim_reduction import *
import dgl
from dgllife.utils import Meter
import numpy as np
import random
from pytorchtools import EarlyStopping
import torch.nn as nn
import random
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

import numpy as np
import torch
from rdkit import RDLogger

from operator import inv
from sklearn.metrics import mutual_info_score
import numpy as np
import pickle
from scipy.stats import entropy

def compute_entropy(feature):
    # Calculate proportions of 1s and 0s
    p_1 = np.sum(feature == 1) / len(feature)
    p_0 = 1 - p_1  # Since it's binary, p(0) = 1 - p(1)
    return entropy([p_1, p_0], base=2)

def calcMutualInformation(data, prefix):
  df = pd.DataFrame()
  row = 0
  for i in range(data.shape[1]-1):
    for j in range(i+1, data.shape[1]):
      mi = mutual_info_score(data.iloc[:, i], data.iloc[:, j])
      df.loc[0] = [i, j, mi]
  df.to_csv(f"{prefix}_mutual_info.csv")

def dimReduction(data, name, threshold):
  entropies = [compute_entropy(data.iloc[:, i]) for i in range(data.shape[1])]
  print(f"min {min(entropies)}")
  print(f"max {max(entropies)}")
  print(f"mean {np.mean(entropies)}")
  print(f"25% {np.percentile(entropies, 25)}")
  print(f"50% {np.percentile(entropies, 50)}")
  print(f"75% {np.percentile(entropies, 75)}")
  #threshold = 0.1
  c = len(list(filter(lambda x: x < threshold, entropies)))
  print(f"c {c}/{len(entropies)}")
 
  #breakpoint()
  indices = np.where(np.array(entropies) > threshold)[0]
  #remove = np.where(np.array(entropies) == threshold)[0]

  #unique_df = pd.DataFrame(remove, columns=['j'])
  # Save to CSV
  #unique_df.to_csv(f"{name}_remove_entropy.csv", index=False)

  return data.iloc[:, indices]

def deDupeWithMutualInformation(data, name):
  featureToRemove = pd.read_csv(f"{name}_remove_features.csv")

  featureToRemove = featureToRemove.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

  data = data.drop(data.columns[featureToRemove['j']], axis=1)

  return data

Count = 1564
smiles = pd.read_csv('./Data/smiles.csv')
bigraphs = []
node_featurizer = CanonicalAtomFeaturizer()
edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
for smile in smiles['smiles']:
    bigraphs.append(
        smiles_to_bigraph(
            smile,
            add_self_loop=True,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            canonical_atom_order=False,
        ))
bigraphs = bigraphs[:Count]
#breakpoint()
cdk = pd.read_csv('/content/drive/MyDrive/git/DCAMCP/Data/feature/CDKextended.csv')
sub = pd.read_csv('/content/drive/MyDrive/git/DCAMCP/Data/feature/Substructure.csv')
#maccs = pd.read_csv('/content/drive/MyDrive/git/DCAMCP/MACCS.csv')
estate = pd.read_csv('/content/drive/MyDrive/git/DCAMCP/EState.csv')
#pubchem = pd.read_csv('/content/drive/MyDrive/git/DCAMCP/PubChem.csv')
#breakpoint()
cdk = cdk.iloc[:Count,1:]
sub = sub.iloc[:Count,1:]
#maccs = maccs.iloc[:Count,1:]
estate = estate.iloc[:Count,1:]
#pubchem = pubchem.iloc[:Count,1:]
print("cdk")
#cdk = dimReduction(cdk)
cdk = deDupeWithMutualInformation(cdk, "cdk")
cdk = dimReduction(cdk, "cdk", 0.0)
#calcMutualInformation(cdk, "cdk")
print("sub")
#sub = dimReduction(sub)
#calcMutualInformation(sub, "sub")
sub = deDupeWithMutualInformation(sub, "sub")
sub = dimReduction(sub, "sub", 0.0)
print("estate")
#0.03 50%
#0.3 75%
estate = dimReduction(estate, "estate", 0.0)

#calcMutualInformation(estate, "estate")

fp_con = pd.concat([cdk, sub, estate], axis=1).astype(float)
print(fp_con.shape)
# Load the pickle file
grover_large_fp_file_path = '/content/drive/MyDrive/git/DCAMCP/Data/grover_large_fp.pkl'

with open(grover_large_fp_file_path, 'rb') as file:
    grover_large_fp = pickle.load(file)

grover_large_fp = grover_large_fp.drop(columns=["smiles"])
grover_large_fp = grover_large_fp.iloc[:Count]
#breakpoint()
#fp_con = pd.concat([fp_con, grover_large_fp], axis=1)
fp = np.array(fp_con).tolist()
grover_large_fp = np.array(grover_large_fp).tolist()

labels = []
for i in smiles['Class']:
    labels.append(i)
labels = labels[:Count]
#breakpoint()

def setup_seed(seed):
    random.seed((seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class Mydataset(Dataset):
    def __init__(self, train_features, train_labels, fp, gfp):
        #breakpoint()
        self.x_data = train_features
        self.y_data = train_labels
        self.fp = fp
        self.gfp = gfp
        #breakpoint()
        #print("~~~~~~~~")
        #print(len(train_labels))
        #print(len(gfp))
        #breakpoint()
        self.len = len(train_labels)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.fp[index], self.gfp[index]

    def __len__(self):
        return self.len

    def get_collate_fn(self):
        def _collate(data):
            #breakpoint()
            graphs, labels, fp, gfp = map(list, zip(*data))
            batched_graph = dgl.batch(graphs)
            return batched_graph, torch.tensor(labels), torch.tensor(fp), torch.tensor(gfp)
        return _collate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_kfold_data(k, i, X, y, z, gfp):
    #breakpoint()
    fold_size = len(X) // k
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid, z_valid, gfp2 = X[val_start:val_end], y[val_start:val_end], z[val_start:val_end], gfp[val_start:val_end]
        X_train = X[0:val_start] + X[val_end:]
        y_train = y[0:val_start] + y[val_end:]
        z_train = z[0:val_start] + z[val_end:]
        gfp = gfp[0:val_start] + gfp[val_end:]
    else:
        X_valid, y_valid, z_valid, gfp2 = X[val_start:], y[val_start:], z[val_start:], gfp[val_start:]
        X_train = X[0:val_start]
        y_train = y[0:val_start]
        z_train = z[0:val_start]
        gfp = gfp[0:val_start]

    return X_train, y_train, z_train, gfp, X_valid, y_valid, z_valid, gfp2

def traink(kth, model,X_train, y_train, z_train, gfp_train, X_val, y_val, z_val, gfp_val, BATCH_SIZE, learning_rate, TOTAL_EPOCHS):

    data = Mydataset(X_train, y_train, z_train, gfp_train)
    collate = data.get_collate_fn()
    #breakpoint()
    train_loader = DataLoader(data, BATCH_SIZE,collate_fn=collate, shuffle=True)
    val_loader = DataLoader(Mydataset(X_val, y_val, z_val, gfp_val), BATCH_SIZE,collate_fn=collate, shuffle=True)
    model=model.to(device)

    # This one uses raw logits scores
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    # This one uses probabilty outputs
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor= 0.5)
    early_stopping = EarlyStopping(patience = 100, verbose=True)
    bestVal = 0

    losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    SE = []
    SP = []
    PR = []
    F1 = []
    AUC = []
    AUC_curve = []
    FPR = []
    TPR = []

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        correct = 0
        y_scores = []
        y_trues = []
        train_loss = 0
        #breakpoint()
        correct = 0
        for i, (features, labels, fp, gfp) in enumerate(train_loader):
            #breakpoint()
            features = features.to(device)
            fp = fp.to(device)
            gfp = gfp.to(device)
            labels = labels.to(device).float()

            # This one is for CrossEntropyLoss
            # labels = torch.eye(2).index_select(dim=0, index=labels).to(device)

            optimizer.zero_grad()
            logits = model(features, fp, gfp)

            # CrossEntropyLoss
            # loss = criterion(logits,labels)

            # BCEWithLogitsLoss
            # 2 logits output
            # logit_for_class_1 = logits[:, 1]
            # 1 logits output
            logit_for_class_1 = logits

            loss = criterion(logit_for_class_1,labels.view(-1, 1))

             # only for BCELoss
            '''
            probs = torch.sigmoid(logits)    # Get predicted class (0 or 1)
            labels = labels.view(-1, 1).float().to(device)
            loss = criterion(probs,labels)

            preds = (probs >= 0.5).long()

            TP = ((preds == 1) & (labels == 1)).sum().item()
            FP = ((preds == 1) & (labels == 0)).sum().item()
            FN = ((preds == 0) & (labels == 1)).sum().item()
            TN = ((preds == 0) & (labels == 0)).sum().item()

            correct += TP + TN
            '''

            train_loss += loss * len(labels)      
            loss.backward()
            #breakpoint()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())

            # CrossEntropyLoss
            #correct += torch.sum(
            #    torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item()

            # BCEWithLogitsLoss
            # Apply softmax to get probabilities
            # 2 logits output
            '''
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            '''
            # 1 logit
            probs = torch.sigmoid(logits)  # Probabilities in [0, 1]
            # Use a threshold of 0.5 to classify
            preds = (probs >= 0.5).float()
            labels = labels.view(-1, 1)

            TP = ((preds == 1) & (labels == 1)).sum().item()
            TN = ((preds == 0) & (labels == 0)).sum().item()
            FP = ((preds == 1) & (labels == 0)).sum().item()
            FN = ((preds == 0) & (labels == 1)).sum().item()
            
            correct += TP + TN

        accuracy = 100. * correct / len(X_train)
        train_loss = train_loss / len(X_train)
        print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'
              .format(epoch + 1, train_loss, correct, len(X_train), accuracy))
        train_acc.append(accuracy)

        model.eval()
        val_loss = 0
        correct = 0
        eval_meter = Meter()
        TP, TN, FP, FN = 0, 0, 0, 0
        with torch.no_grad():
            for i, (features,labels,fp, gfp) in enumerate(val_loader):

                features = features.to(device)
                fp = fp.to(device)
                gfp = gfp.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
      
                logits= model(features, fp, gfp)

                # for CrossEntropyLoss
                # labels_1 = torch.eye(2).index_select(dim=0, index=labels).to(device)
                # loss = criterion(logits, labels_1).item()
                
                # BCEWithLogitsLoss
                # 2 logits
                # logit_for_class_1 = logits[:, 1]
                # 1 logit
                logit_for_class_1 = logits

                loss = criterion(logit_for_class_1,labels.view(-1, 1).float())
                # only for BCELoss
                '''
                probs = torch.sigmoid(logits)    # Get predicted class (0 or 1)
                labels_1 = labels.view(-1, 1).float().to(device)

                loss = criterion(probs, labels_1).item()      
                y_score = probs
                preds = (probs >= 0.5).long()

                TP = ((preds == 1) & (labels_1 == 1)).sum().item()
                FP = ((preds == 1) & (labels_1 == 0)).sum().item()
                FN = ((preds == 0) & (labels_1 == 1)).sum().item()
                TN = ((preds == 0) & (labels_1 == 0)).sum().item()

                correct += TP + TN
                '''

                # CrossEntropyLoss
                # y_score = logits[:,1].view(-1,1)

                # BCEWithLogitsLoss
                # Apply softmax to get probabilities
                # 2 logit
                # probs = torch.nn.functional.softmax(logits, dim=1)
                # 1 logit 
                probs = torch.sigmoid(logits)  # Probabilities in [0, 1]
                # Extract the probability for class 1 (positive class)
                # 2 logits
                # y_score = probs[:, 1].detach().cpu().numpy()
                # 1 logits
                y_score = probs.detach().cpu().numpy()
                
                y_true = labels.view(-1,1)
                y_scores.extend(y_score)
                y_trues.extend(y_true.cpu().numpy())

                val_loss += loss * len(labels)

                #cross entropy loss
                '''         
                for idx, i in enumerate(logits):
                    if torch.argmax(i) == torch.argmax(labels_1[idx]) and torch.argmax(i) == 1:
                        TP += 1
                    if torch.argmax(i) == torch.argmax(labels_1[idx]) and torch.argmax(i) == 0:
                        TN += 1
                    if torch.argmax(i) != torch.argmax(labels_1[idx]) and torch.argmax(i) == 1:
                        FP += 1
                    if torch.argmax(i) != torch.argmax(labels_1[idx]) and torch.argmax(i) == 0:
                        FN += 1
                            
                correct += torch.sum(
                    torch.argmax(logits, dim=1) == torch.argmax(labels_1, dim=1)).item()
                '''

                #BCEWithLogitsLoss
                # 2 logits
                # preds = torch.argmax(logits, dim=1)
                
                # Use a threshold of 0.5 to classify
                # 1 logit
                preds = (probs >= 0.5).float()
                labels = labels.view(-1, 1)

                TP = ((preds == 1) & (labels == 1)).sum().item()
                TN = ((preds == 0) & (labels == 0)).sum().item()
                FP = ((preds == 1) & (labels == 0)).sum().item()
                FN = ((preds == 0) & (labels == 1)).sum().item()
                correct += TP + TN
  
        sp = TN / (TN + FP)
        se = TP / (TP + FN)
        pr  = TP / (TP + FP)
        #auc = np.mean(eval_meter.compute_metric('roc_auc_score'))
        auc = roc_auc_score(y_trues, y_scores)
        f1 = 2 * pr * se/(pr + se)
        fpr, tpr, _ = roc_curve(y_trues, y_scores)

        val_loss = val_loss / len(X_val)
        val_losses.append(val_loss)
        SE.append(se)
        SP.append(sp)
        PR.append(pr)
        AUC.append(auc)
        F1.append(f1)
        FPR.append(fpr)
        TPR.append(tpr)

        accuracy = 100. * correct / len(X_val)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n AUC: {:.3f}'
              ', SE: {:.3f}, SP: {:.3f}, precision: {:.3f}, F1: {:.3f}\n'.format(
        val_loss, correct, len(X_val), accuracy, auc, se, sp, pr, f1))

        if bestVal < se:
          bestVal = se
          print(f"model saved best recall {bestVal} auc {auc} precision {pr} sp {sp}")
          torch.save(model, f"mymodel_full_{kth}.pth")
          df = pd.DataFrame(columns = ['fpr', 'tpr'])
          df['fpr'] = fpr
          df['tpr'] = tpr
          df.to_csv('roc_{:.2f}.csv'.format(auc))
          plt.figure(figsize=(8, 6))
          plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
          plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier baseline
          plt.xlabel('False Positive Rate (FPR)')
          plt.ylabel('True Positive Rate (TPR)')
          plt.title('ROC Curve')
          plt.legend(loc='lower right')
          plt.savefig('roc_{:.2f}.png'.format(auc), dpi = 1200, bbox_inches='tight')
          plt.close()

        val_acc.append(accuracy)

        #optimise by auc
        #scheduler.step(auc)
        #early_stopping(auc, model)
        scheduler.step(se)
        early_stopping(se, model)

        if early_stopping.early_stop:
            print("Early stopping")

            break


    return losses, val_losses, train_acc, val_acc, AUC, SE, SP, PR, F1, FPR, TPR

def k_fold(k, X_train, y_train, z_train, gfp, num_epochs=3, learning_rate=0.0001, batch_size=16):
    #breakpoint()
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    SE_sum, SP_sum, AUC_sum, pr_sum, f1_sum = 0,0,0,0,0
    bestAuc = 0

    for i in range(k):
        print('*' * 25, 'Fold', i + 1, '*' * 25)
        data = get_kfold_data(k, i, X_train, y_train, z_train, gfp)
        #breakpoint()
        fps = torch.tensor(z_train)
        #breakpoint()
        Ind = fps.shape[-1]
        gfpLen = len(gfp[0][0])
        graph = X_train[0]
        n_feats = graph.ndata["h"].shape[1]

        '''
        print(f"fea {n_feats}")
        print(f"ind {Ind}")
        print(f"gfplen {gfpLen}")
        '''
       
        model = Mymodel(n_feats = n_feats, fp = Ind, gfp = gfpLen)
        train_loss, val_loss, train_acc, val_acc, AUC, SE, SP, PR, F1, FPR, TPR= traink(i, model, *data, batch_size, learning_rate, num_epochs)
       
        index = SE.index(max(SE))
        print("-----------------------------------")
        print('best recall: {:.3f}, precision: {:.3f}, auc: {:.3f}, sp: {:.3f}%\n'.format(SE[index], PR[index], AUC[index], SP[index]))
        print('train_loss:{:.5f}, train_acc:{:.3f}%'.format(train_loss[index], train_acc[index]))
        print('valid loss:{:.5f}, valid_acc:{:.3f}%\n'.format(val_loss[index], val_acc[index]))
        
        train_loss_sum += train_loss[index]
        valid_loss_sum += val_loss[index]
        train_acc_sum += train_acc[index]
        valid_acc_sum += val_acc[index]

        SE_sum += SE[index]
        SP_sum += SP[index]
        AUC_sum += AUC[index]
        f1_sum += F1[index]
        pr_sum += PR[index]

        if AUC[index]>bestAuc:
          bestAuc = AUC[index]

    print('\n', '#' * 10, 'Final k-fold results', '#' * 10)

    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / k, train_acc_sum / k))
    print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / k, valid_acc_sum / k))
    print('average valid AUC:{:.3f}, average valid SE:{:.3f}, average valid SP:{:.3f}, average F1:{:.3f}, average precision:{:.3f}'
          .format(AUC_sum / k,SE_sum / k,SP_sum / k, f1_sum/k, pr_sum/k))
    #print(f"best AUC {bestAuc}")

    return

if __name__ == '__main__':
    setup_seed(2)
    k_fold(5,bigraphs,labels,fp,grover_large_fp,num_epochs=200,learning_rate=0.001 ,batch_size=256)
