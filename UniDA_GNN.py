from argparse import ArgumentParser
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import itertools
from torch_geometric.utils import subgraph
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import GAT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acm')
parser.add_argument("--target", type=str, default='dblp')
parser.add_argument("--feature_layer", type=int, default=0, choices=[0, 1, 2], help='GCN: 0, GraphSAGE: 1, GAT: 2')
parser.add_argument("--seed", type=int,default=200)
parser.add_argument("--noise_prob", type=float,default=0.2)
parser.add_argument("--hidden_dim", type=int, default=1000)
parser.add_argument("--encoder_dim", type=int, default=128)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--theta", type=float, default=-0.36)
parser.add_argument("--alpha", type=float, default=0.12)
parser.add_argument("--tau", type=float, default=0.1)
parser.add_argument("--teacher_lambda", type=float, default=0.2)

args = parser.parse_args()
feature_layer = "GCN" if args.feature_layer == 0 else "GraphSAGE" if args.feature_layer == 1 else "GAT"
seed = args.seed
noise_prob = args.noise_prob
hidden_dim = args.hidden_dim
encoder_dim = args.encoder_dim
epochs = args.epochs
theta = args.theta
alpha = args.alpha
tau = args.tau
teacher_lambda = args.teacher_lambda

id = "source: {}, target: {}, feature_layer {}, seed: {}, noise_prob: {},\
    hidden_dim: {}, encoder_dim: {}, epochs: {}, theta: {}, alpha: {}, tau: {}, lambda: {}"\
    .format(args.source, args.target, feature_layer, seed, noise_prob, \
            hidden_dim, encoder_dim, epochs, theta, alpha, tau, teacher_lambda)

print(id)

def class_list(common, source_private, target_private):
    common_classes = [i for i in range(common)]
    source_classes = common_classes + [i + common for i in range(source_private)]
    target_classes = common_classes + [i + common + source_private for i in range(target_private)]
    return source_classes, target_classes

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
source_classes, target_classes = class_list(common=6, source_private=0, target_private=0)

def add_mask(data):
    data_len = data.y.shape[0]
    random_node_indices = np.random.permutation(data_len)
    training_size = int(len(random_node_indices) * 0.7)
    val_size = int(len(random_node_indices) * 0.1)
    train_node_indices = random_node_indices[:training_size]
    val_node_indices = random_node_indices[training_size:training_size + val_size]
    test_node_indices = random_node_indices[training_size + val_size:]

    train_masks = torch.zeros([data_len], dtype=torch.bool)
    train_masks[train_node_indices] = 1
    val_masks = torch.zeros([data_len], dtype=torch.bool)
    val_masks[val_node_indices] = 1
    test_masks = torch.zeros([data_len], dtype=torch.bool)
    test_masks[test_node_indices] = 1

    data.train_mask = train_masks
    data.val_mask = val_masks
    data.test_mask = test_masks

def set_UniDA(data, type):
    classes = source_classes if type == "source" else target_classes
    class_mask = np.isin(data.y, classes)
    nodes = [i for i, mask in enumerate(class_mask) if mask]
    edge_index, _ = subgraph(nodes, data.edge_index, relabel_nodes=True)
    data.x = data.x[class_mask]
    data.edge_index = edge_index
    data.y = data.y[class_mask]
    add_mask(data)
    return data

def add_uniform_noise(label, prob):
    classes = torch.unique(label)
    num_label_classes = len(classes)
    for i in range(len(label)):
        p = np.full((num_label_classes, ), prob/(num_label_classes-1))
        p[(classes == label[i])] = 1 - prob
        label[i] = np.random.choice(classes, p=p)
    return label

dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = set_UniDA(dataset[0], type="source")
source_data.y = add_uniform_noise(source_data.y, noise_prob)
add_mask(dataset[0])
print(source_data)

dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = set_UniDA(dataset[0], type="target")
add_mask(dataset[0])
print(target_data)

source_data = source_data.to(device)
target_data = target_data.to(device)

class GNN(torch.nn.Module):
    def __init__(self, base_model=None, type="gcn", **kwargs):
        super(GNN, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]


        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv

        self.conv_layers = nn.ModuleList([
            model_cls(dataset.num_features, hidden_dim,
                     weight=weights[0],
                     bias=biases[0],
                      **kwargs),
            model_cls(hidden_dim, encoder_dim,
                     weight=weights[1],
                     bias=biases[1],
                      **kwargs)
        ])

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x

class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = epochs

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1
    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

class TargetDomainMLP(nn.Module):
    def __init__(self, in_feature, hidden_size, feature_size):
        super(TargetDomainMLP, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, feature_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        y = self.ad_layer2(x)
        return y
    
def generate_model():
    model = []
    if feature_layer == "GCN":
        adj_encoder = GNN(type="gcn").to(device)
        ppmi_encoder = GNN(base_model=adj_encoder, type="ppmi", path_len=10).to(device)
        att_model = Attention(encoder_dim).to(device)
        model.extend([adj_encoder, ppmi_encoder, att_model])
    elif feature_layer == "GraphSAGE":
        graphSAGE_encoder = GraphSAGE(
            in_channels=dataset.num_features,
            hidden_channels=hidden_dim,
            num_layers=2,
            out_channels=encoder_dim,
        ).to(device)
        model.append(graphSAGE_encoder)
    else:
        gat_encoder = GAT(
            in_channels=dataset.num_features,
            hidden_channels=hidden_dim,
            num_layers=2,
            out_channels=encoder_dim,
        ).to(device)
        model.append(gat_encoder)
    cls_model = nn.Sequential(
        nn.Linear(encoder_dim, len(source_classes)),
    ).to(device)
    model.append(cls_model)
    return model

loss_func = nn.CrossEntropyLoss().to(device)
models = generate_model()
cls_model = models[3] if feature_layer == "GCN" else models[1]
ad_net = AdversarialNetwork(encoder_dim * len(source_classes), 30).to(device)
target_mlp = TargetDomainMLP(dataset.num_features, hidden_dim, encoder_dim).to(device)
models.extend([ad_net, target_mlp])
params = itertools.chain(*[model.parameters() for i, model in enumerate(models) if i != 1])
optimizer = torch.optim.Adam(params, lr=3e-3)

model_teacher = generate_model()
model_teacher2 = generate_model()

def gcn_encode(model, data, cache_name, mask=None):
    encoded_output = model(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def ppmi_encode(model, data, cache_name, mask=None):
    encoded_output = model(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def encode(model, data, cache_name, mask=None):
    if feature_layer == "GCN":
        gcn_output = gcn_encode(model[0], data, cache_name, mask)
        ppmi_output = ppmi_encode(model[1], data, cache_name, mask)
        outputs = model[2]([gcn_output, ppmi_output])
    else:
        outputs = model[0](data.x, data.edge_index)
    return outputs

def predict(data, cache_name, mask=None, pruning_mask=None):
    encoded_output = encode(models, data, cache_name, mask)
    if pruning_mask is not None:
        encoded_output[:,pruning_mask] = 0
    logits = cls_model(encoded_output)
    return logits

def outlier(score):
    return score < theta

def Entropy(input_):
    if len(input_) == 0:
        return 0
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=-1)
    return entropy

def scoring(predict_prob):
    predict_prob.detach()
    confidence, _ = predict_prob.max(dim=-1)
    entropy = Entropy(predict_prob)
    entropy_norm = entropy / np.log(predict_prob.size(-1))
    score = confidence - entropy_norm
    score = score.detach()
    return score

def separating(score, theta=0.5, alpha=0.2):
    common_mask = (score - theta > alpha)
    unknown_mask = (theta - score > alpha)
    return common_mask, unknown_mask

def evaluate(logits, labels, cache_name):
    corrects = 0
    prob = logits.softmax(dim=-1)
    pred = prob.argmax(dim=-1)
    outlier_mask = outlier(scoring(prob))
    for i in range(len(labels)):
        if labels[i] in source_classes:
            corrects += ((not outlier_mask[i]) and pred[i].eq(labels[i]))
        else:
            corrects += outlier_mask[i]
    accuracy = float(corrects) / len(labels)
    return accuracy

def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    logits = predict(data, cache_name, mask)
    labels = data.y if mask is None else data.y[mask]
    accuracy = evaluate(logits, labels, cache_name)
    return accuracy

def CDAN(input_list, ad_net, source_len, target_len, entropy=None, coeff=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
    ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    dc_target = torch.from_numpy(np.array([[1]] * source_len + [[0]] * target_len)).float().to(device)
    entropy.register_hook(grl_hook(coeff))
    entropy = 1.0+torch.exp(-entropy)
    source_mask = torch.ones_like(entropy).to(device)
    source_mask[target_len:] = 0
    source_weight = entropy*source_mask
    target_mask = torch.ones_like(entropy).to(device)
    target_mask[0:source_len] = 0
    target_weight = entropy*target_mask
    epsilon = 1e-9
    weight = source_weight / (torch.sum(source_weight).detach().item() + epsilon) + \
            target_weight / (torch.sum(target_weight).detach().item() + epsilon)
    return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / (torch.sum(weight).detach().item() + epsilon)
    
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=epochs):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def h(z1, z2):
    sim = torch.matmul(z1, z2)
    return torch.exp(sim / tau)

def intra_contrastive_loss(encoded_target, encoded_target_mlp):
    encoded_target = F.normalize(encoded_target)
    encoded_target_mlp = F.normalize(encoded_target_mlp)
    h1 = h(encoded_target, encoded_target_mlp.T)
    h2 = h1.sum(dim=1)
    h1 = h1.diagonal()
    h3 = h(encoded_target_mlp, encoded_target_mlp.T)
    h3 = h3.sum(dim=1) - h3.diagonal()
    loss = torch.log(h1 / (h2 + h3)).sum()
    return -(loss / len(encoded_target))

min_percentile_w = 10
max_percentile_w = 99
min_percentile_o = 10
max_percentile_o = 99

def OPNP():
    logits = predict(source_data, "source")
    energy_score = torch.log(torch.exp(logits).sum(dim=-1)).mul(-1 / len(logits)).sum()
    optimizer.zero_grad()
    energy_score.backward()
    grads = torch.tensor([]).to(device)
    for param in cls_model.parameters():
        grads = torch.cat((grads, param.grad.view(-1).abs()))
    grads = grads.to("cpu").numpy()
    min_threshold_w = np.percentile(grads, min_percentile_w)
    max_threshold_w = np.percentile(grads, max_percentile_w)
    min_threshold_o = np.percentile(grads, min_percentile_o)
    max_threshold_o = np.percentile(grads, max_percentile_o)
    fc_layer_param = next(cls_model.parameters()).clone()
    fc_layer_param = fc_layer_param.sum(dim=0).view(-1)
    for param in cls_model.parameters():
        with torch.no_grad():
            param[(param.grad < min_threshold_w) | (param.grad > max_threshold_w)] = 0.0
    pruning_mask = (fc_layer_param < min_threshold_o) | (fc_layer_param > max_threshold_o)
    logits = predict(target_data, "target", pruning_mask=pruning_mask)
    accuracy = evaluate(logits, target_data.y, "target")
    print("After OPNP target_acc: {}".format(accuracy))
    return


def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()
    
    encoded_source = encode(models, source_data, "source")
    encoded_target = encode(models, target_data, "target")

    source_logits = cls_model(encoded_source)
    target_logits = cls_model(encoded_target)
    
    loss = loss_func(source_logits, source_data.y)

    source_predict_prob = source_logits.softmax(dim=1)
    target_predict_prob = target_logits.softmax(dim=1)

    target_score = scoring(target_predict_prob)
    target_common_mask, target_private_mask = separating(target_score, theta=theta, alpha=alpha)

    feature = torch.cat((encoded_source, encoded_target[target_common_mask]), 0)
    output = torch.cat((source_predict_prob, target_predict_prob[target_common_mask]), 0)

    loss += epoch / epochs * CDAN([feature, output], ad_net, len(encoded_source), target_common_mask.sum(), 
                                      entropy=Entropy(output), coeff=calc_coeff(epoch))
    
    targat_private_probs = target_predict_prob[target_private_mask]
    loss_entropy = -torch.mean(Entropy(targat_private_probs))

    loss += (epoch / epochs * loss_entropy)

    if tau > 0:
        encoded_target_mlp = target_mlp(target_data.x)
        loss += (epoch / epochs * intra_contrastive_loss(encoded_target, encoded_target_mlp))

    if teacher_lambda > 0:
        if epoch % 2 == 0:
            teacher = model_teacher
        else:
            teacher = model_teacher2

        for model in teacher:
            model.train()

        with torch.no_grad():
            encoded_source_t = encode(teacher, source_data, "source")
            encoded_target_t = encode(teacher, target_data, "target")

            feature_t = torch.cat((encoded_source_t, encoded_target_t), 0)
            if feature_layer == "GCN":
                output_t = teacher[3](feature_t)
            else:
                output_t = teacher[1](feature_t)
            pred_t = nn.Softmax(dim=-1)(output_t)

        output_s = torch.cat((source_logits, target_logits), 0).softmax(dim=-1)

        loss += epoch / epochs * torch.sum((-pred_t * torch.log(output_s + 1e-9))) / len(output_s) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            alpha_teacher = min(1 - 1 / (epoch + 1), 0.99)
            for i in range(len(teacher)):
                for ema_param, param in zip(teacher[i].parameters(), models[i].parameters()):
                    ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
for epoch in range(epochs):
    train(epoch)
    source_correct = test(source_data, "source", source_data.test_mask)
    target_correct = test(target_data, "target")
    print("Epoch: {}, source_acc: {}, target_acc: {}".format(epoch + 1, source_correct, target_correct))
    if target_correct > best_target_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_epoch = epoch
print("=============================================================")
line = "{} - Epoch: {}, best_source_acc: {}, best_target_acc: {}"\
    .format(id, best_epoch + 1, best_source_acc, best_target_acc)

print(line)
OPNP()
