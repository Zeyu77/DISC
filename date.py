import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torchvision.transforms as transforms
from loguru import logger
import torch.nn.functional as F
import itertools
from model_loader import load_model
from evaluate import mean_average_precision
from torch.nn import Parameter
from sklearn.cluster import KMeans
from shc import SHC
from dhc import DHC
from torch.distributions import MultivariateNormal

def train(train_s_dataloader,
          train_t_dataloader,
          query_dataloader,
          retrieval_dataloader,
          code_length,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_class,
          evaluate_interval,
          tag,
          training_source,
          training_target,
          num_features,
          max_iter_target,
          gpu_number,
          save_source_statistics,
          alpha,
          eta,
          ):

    if training_source:
        logger.info('Training on source starts')
        model = load_model(arch, code_length).to(device)

        parameter_list = model.parameters()
        optimizer = optim.SGD(parameter_list, lr=lr, momentum=0.9, weight_decay=1e-5)

        criterion_s = Source_Loss()

        source_labels = extract_source_labels(train_s_dataloader, num_class, verbose)
        S_s = torch.mm(source_labels, source_labels.t())
        S_s[S_s == 0] = -1
        S_s = S_s.to(device)
        model.train()

        for epoch in range(max_iter):
            for i, (data_s,  _, index) in enumerate(train_s_dataloader):

                data_s = data_s.to(device)
                optimizer.zero_grad()
                _, code_s = model(data_s)
                H_s = code_s @ code_s.t() / code_length
                source_targets = S_s[index, :][:, index]
                loss_s = criterion_s(H_s, source_targets)

                total_loss = loss_s
                total_loss.backward()
                optimizer.step()

            
            # Print log
            logger.info('[Epoch:{}/{}][loss:{:.4f}]'.format(epoch+1, max_iter, total_loss.item()))


            # Evaluate
            if epoch == 49:
                mAP = evaluate(model,
                                query_dataloader,
                                retrieval_dataloader,
                                code_length,
                                device,
                                topk,
                                save = False,
                                )
                logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                    epoch+1,
                    max_iter,
                    mAP,
                ))

                if epoch == 49:
                    if not os.path.exists("./checkpoint"):
                        os.makedirs("./checkpoint")
                    torch.save({'iteration': epoch+1,
                                'model_state_dict': model.state_dict(),
                                }, os.path.join('checkpoint', 'resume_{}.t'.format(epoch+1)))
        # Evaluate and save
        mAP = evaluate(model,
                    query_dataloader,
                    retrieval_dataloader,
                    code_length,
                    device,
                    topk,
                    save=False,
                    )

        logger.info('Training on source finished, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))

    if save_source_statistics:
        logger.info('Saving source statistics')
        model_source = load_model(arch, code_length).to(device)

        saved_state_dict_ = torch.load('checkpoint/resume_{}.t'.format(50),map_location=torch.device("cuda:%d" % gpu_number))

        model_source.load_state_dict(saved_state_dict_['model_state_dict'])

        source_features, source_labels = generate_source_statistics(model_source, train_s_dataloader, num_features, num_class, device)
        source_labels = source_labels.argmax(dim=1)
        feature_dim = num_features

        mean_vectors = torch.zeros(num_class, feature_dim)
        cov_matrices = torch.zeros(num_class, feature_dim, feature_dim)


        for c in range(num_class):
            indices = (source_labels == c).nonzero(as_tuple=True)[0]

            mean_vector = source_features[indices].mean(dim=0)
            mean_vectors[c] = mean_vector

            cov_matrix = (source_features[indices] - mean_vector).T @ (source_features[indices] - mean_vector) / \
                         indices.shape[0]
            cov_matrices[c] = cov_matrix

        torch.save(torch.FloatTensor(mean_vectors), 'means.pt')
        torch.save(torch.FloatTensor(cov_matrices), 'covs.pt')
        print(torch.isnan(mean_vectors).any())
        print(torch.isinf(mean_vectors).any())
        print((mean_vectors > 1e10).any())

        logger.info('Saving source feature centers--Done')

    if training_target:
        logger.info('Training on target starts')
        model_target = load_model(arch, code_length).to(device)


        saved_state_dict = torch.load('checkpoint/resume_{}.t'.format(50), map_location=torch.device("cuda:%d" % gpu_number))
        model_target.load_state_dict(saved_state_dict['model_state_dict'])


        model_target.hash_layer_source.load_state_dict(model_target.hash_layer.state_dict())

        # Freeze the parameters of hash_layer_source
        for param in model_target.hash_layer_source.parameters():
            param.requires_grad = False

        parameter_list_target = model_target.parameters()
        optimizer_target = optim.SGD(parameter_list_target, lr=lr, momentum=0.9, weight_decay=1e-5)


        criterion_SHC = SHC(temperature=0.07)


        logger.info('Construct distilled graph structure')

        means = torch.load('means.pt')

        features_target = extract_features(model_target, train_t_dataloader,num_features, device,verbose)
        features_ = F.normalize(features_target, dim=-1)


        S_ = features_ @ features_.t()
        threshold_s1 = torch.kthvalue(S_.flatten(), int(S_.numel() * (1-alpha))).values

        S = (S_>=threshold_s1)*1.0 + (S_<threshold_s1)*0.0


        means_np = means.numpy()
        Y, M = generate_cluster_graph(features_target,means_np,num_class)

        D = (S.to(torch.bool) & M.to(torch.bool)).to(torch.float)

        logger.info('done')
        torch.save(Y, 'Y.pt')
        torch.save(D, 'D.pt')


        Y = torch.load('Y.pt')
        means = torch.load('means.pt')
        covs = torch.load('covs.pt')
        D = torch.load('D.pt')
        covs += (1e-5 * torch.eye(covs.shape[1])).unsqueeze(0)

        Y = Y.to(device)
        means = means.to(device)
        covs = covs.to(device)

        model_target.train()

        for epoch in range(max_iter_target):

            running_loss = 0.0
            if (epoch == 0):
                mAP = evaluate(model_target,
                               query_dataloader,
                               retrieval_dataloader,
                               code_length,
                               device,
                               topk,
                               save=False,
                               )
                logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                    epoch + 1,
                    max_iter_target,
                    mAP,
                ))
            for i, (data1,data2, _, index) in enumerate(train_t_dataloader):

                data1 = data1.to(device)
                data2 = data2.to(device)
                batch_size = data1.shape[0]
                optimizer_target.zero_grad()

                _,v = model_target(data1)
                _,v_aug = model_target(data2)

                pseudo_label = Y[index, :]
                class_counts = torch.bincount(pseudo_label.flatten(), minlength=num_class)
                class_counts = class_counts.unsqueeze(1).int()
                h1 = F.normalize(v)
                h2 = F.normalize(v_aug)
                codes = torch.cat([h1.unsqueeze(1), h2.unsqueeze(1)], dim=1)
                mask = D[index, :][:, index]
                loss_shc = criterion_SHC(codes, mask=mask)

                sum = class_counts[class_counts >= 1].sum()
                v_filtered = torch.empty(sum, code_length).to(device)
                pseudo_label_filtered = torch.empty(sum, 1, dtype=torch.long).to(device)
                class_counts_filtered = torch.zeros_like(class_counts).to(device)

                m = 0
                for c in range(num_class):
                    if class_counts[c] >= 1:
                        idx = (pseudo_label == c).nonzero(as_tuple=True)[0]
                        v_filtered[m:m + class_counts[c]] = v[idx]
                        pseudo_label_filtered[m:m + class_counts[c]] = pseudo_label[idx]
                        class_counts_filtered[c] = class_counts[c]
                        m += class_counts[c]

                dhc = DHC(kernel_num=(5, 5),
                       kernel_mul=(2, 2),
                       num_layers=1,
                       num_classes=num_class,
                       intra_only=False)

                samples = []
                for i in range(num_class):
                    mean = means[i]
                    cov = covs[i]
                    dist = MultivariateNormal(mean, cov)
                    class_samples = dist.sample((class_counts_filtered[i].item(),))
                    samples.append(class_samples)

                samples = torch.cat(samples).to(device)
                source_hash_codes = model_target.hash_layer_source(samples)
                number_selected_classes = class_counts_filtered[class_counts_filtered != 0]
                loss_dhc = dhc.forward(v_filtered, source_hash_codes,number_selected_classes, number_selected_classes)['hdc']
                if loss_dhc is None:
                    continue

                loss_target = loss_shc + loss_dhc * eta
                running_loss += loss_target.item()
                loss_target.backward()
                optimizer_target.step()
            n_batch = len(train_t_dataloader)
            epoch_loss = running_loss / n_batch
            logger.info('[Epoch:{}/{}][loss:{:.4f}]'.format(epoch + 1, max_iter_target, epoch_loss))

            # Evaluate
            if (epoch % evaluate_interval == evaluate_interval - 1):
                mAP = evaluate(model_target,
                               query_dataloader,
                               retrieval_dataloader,
                               code_length,
                               device,
                               topk,
                               save=False,
                               )
                logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                    epoch + 1,
                    max_iter_target,
                    mAP,
                ))

        # Evaluate and save
        mAP = evaluate(model_target,
                    query_dataloader,
                    retrieval_dataloader,
                    code_length,
                    device,
                    topk,
                    save=False,
                    )

        logger.info('Training on target finished, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))

def evaluate(model,query_dataloader, retrieval_dataloader, code_length, device, topk, save):
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)

    # One-hot encode targets

    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
   
    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )

    if save:
        np.save("code/query_code_{}_mAP_{}".format(code_length, mAP), query_code.cpu().detach().numpy())
        np.save("code/retrieval_code_{}_mAP_{}".format(code_length, mAP), retrieval_code.cpu().detach().numpy())
        np.save("code/query_target_{}_mAP_{}".format(code_length, mAP), onehot_query_targets.cpu().detach().numpy())
        np.save("code/retrieval_target_{}_mAP_{}".format(code_length, mAP), onehot_retrieval_targets.cpu().detach().numpy())
    
    model.train()
    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data,  _,index in dataloader:
            data = data.to(device)
            _,outputs= model(data)
            code[index, :] = outputs.sign().cpu()

    return code

def generate_source_statistics(model, dataloader, code_length, num_class, device):
    """
    Generate hash code. Actually is the features

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        labels = torch.zeros(dataloader.dataset.data.shape[0], num_class)
        for data,  label,index in dataloader:
            data = data.to(device)
            outputs,_= model(data)
            code[index, :] = outputs.cpu()
            labels[index, :] = label.float()
    return code, labels

def extract_features(model, dataloader, num_features, device, verbose):
    """
    Extract features.
    """
    model.eval()
    model.set_extract_features(True)
    features_ = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    with torch.no_grad():
        N = len(dataloader)
        for i, (data_1, _,_, index) in enumerate(dataloader):
            if verbose:
                logger.debug('[Batch:{}/{}]'.format(i+1, N))
            data_1 = data_1.to(device)
            features_[index, :],_ = model(data_1)[0].cpu(),model(data_1)[1].cpu()

    model.set_extract_features(False)
    model.train()
    return features_

def extract_source_labels(dataloader, num_class, verbose):
    """
    Extract source_labels.
    """
    labels = torch.zeros(dataloader.dataset.data.shape[0], num_class)
    with torch.no_grad():
        N = len(dataloader)
        for i, (_, label, index) in enumerate(dataloader):
            if verbose:
                logger.debug('[Batch:{}/{}]'.format(i+1, N))
            labels[index, :] = label.float()
    return labels

class Source_Loss(nn.Module):
    def __init__(self):
        super(Source_Loss, self).__init__()

    def forward(self, H, S):
        loss = (S.abs() * (H - S).pow(2)).sum() / (H.shape[0] ** 2)
        return loss

def generate_cluster_graph(features, source_center, Classes):
    features_norm = (features.T / np.linalg.norm(features, axis=1)).T
    source_center_norm = (source_center.T / np.linalg.norm(source_center, axis=1)).T
    kmeans = KMeans(n_clusters=Classes, init=source_center_norm).fit(features_norm)
    A = kmeans.labels_[np.newaxis, :]
    labels = kmeans.labels_.reshape(-1, 1)
    S = ((A - A.T) == 0)
    return torch.tensor(labels).int(), torch.FloatTensor(S)