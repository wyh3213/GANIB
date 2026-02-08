from layers import *
from StarNet import *


class InformationBottleneck(nn.Module):
    def __init__(self):
        super(InformationBottleneck, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(256)

        # Encoder part: mapping input to high dimensional space
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        # Generate mean and variance (logarithmic variance)
        self.vec_mean = nn.Linear(1024, 256)# Mean vector
        self.vec_cov = nn.Linear(1024, 256)# Variance vector
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, pair):
        # Encoder part
        pair = self.dropout(self.bn1(pair))
        pair = self.bn2(F.relu(self.fc1(pair)))
        pair = self.bn2(F.relu(self.fc2(pair)))

        # Parameters for generating Gaussian distribution
        vec_mean, vec_cov = self.bn3(self.vec_mean(pair)), F.softplus(self.bn3(self.vec_cov(pair)) - 5)
        eps = torch.randn_like(vec_cov)         # Sampling noise ε ~ N(0,1)
        jointF = vec_mean + vec_cov * eps       # Generate implicit variable z = μ + σ*ε
        return jointF, vec_mean, vec_cov

class DyT(torch.nn.Module):
    def __init__(self, C, init_alpha):
        super().__init__()
        self.alpha = Parameter(torch.ones(1) * init_alpha)
        self.gamma = Parameter(torch.ones(C))
        self.beta = Parameter(torch.zeros(C))
        self.init_alpha = init_alpha

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.gamma + self.beta


class GANIB(nn.Module):
    def __init__(
            self,
            hops,
            output_dim,
            input_dim,
            pe_dim,
            num_dis,
            num_meta,
            graphformer_layers,
            num_heads,
            hidden_dim,
            ffn_dim,
            dropout_rate,
            GCNII_layers

    ):

        super().__init__()
        self.seq_len = hops + 1
        self.pe_dim = pe_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.graphformer_layers = graphformer_layers
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.num_dis = num_dis
        self.num_meta = num_meta
        self.GCNII_layers = GCNII_layers
        self.convs = nn.ModuleList()

        # Residual Graph Convolution (RGC)
        for i in range(self.GCNII_layers):
            conv = GCN2Conv(channels=int(self.hidden_dim/2), alpha=0.1, theta=1, layer=i + 1)
            self.convs.append(conv)

        # Graph Transformer (GT)
        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)
        encoders = [
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.num_heads)
            for _ in range(self.graphformer_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        ###############################
        decoders = [
            DecoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.num_heads)
            for _ in range(self.graphformer_layers)]
        self.layers2 = nn.ModuleList(decoders)
        self.final_ln2 = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.Linear1 = nn.Linear(int(self.hidden_dim / 2), self.output_dim)
        self.scaling = nn.Parameter(torch.ones(1) * 0.5)

        self.g1a = DyT(256, 2.5)
        self.g1b = DyT(256, 2.5)
        # Multi-layer perceptron (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(192, 256),
            self.g1a,
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.Linear(128, 64)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(192, 256),
            self.g1b,
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 64)
        )
        self.decoder = InnerProductDecoder(self.output_dim, self.dropout_rate, self.num_dis)
        self.apply(lambda module: init_params(module, n_layers=self.graphformer_layers))
        self.residual_norm = nn.LayerNorm(int(self.hidden_dim / 2))
        self.ib = InformationBottleneck()
        self.Linear2 = nn.Linear(192, 256)
        self.Linear3 = nn.Linear(256, 192)

        self.Linear4 = nn.Linear(64, 192)

        self.starnet3 = StarNet(32, [1, 2, 6, 2], 4, drop_path_rate=0.1, num_classes=64)  #data1
        self.starnet4 = StarNet(32, [1, 2, 6, 2], 4, drop_path_rate=0.1, num_classes=64)  #data1
        # self.starnet3 = StarNet(32, [2, 2, 8, 4], 4, drop_path_rate=0.1, num_classes=64)  #data2
        # self.starnet4 = StarNet(32, [2, 2, 8, 4], 4, drop_path_rate=0.1, num_classes=64)  #data2

    def forward(self, processed_features, dis_data, meta_data):
        # Residual graph convolution for disease similarity network coding
        x_0_dis = dis_data.x
        x_dis = x_0_dis
        for conv in self.convs:
            x_dis = conv(x_dis, x_0_dis, dis_data.edge_index)
        x_dis = self.Linear4(x_dis)
        x_dis = x_dis.view(-1, 3, 8, 8)
        x_dis = self.starnet3(x_dis)

        # Residual graph convolution for metabolite similarity network coding
        x_0_meta = meta_data.x
        x_meta = x_0_meta
        for conv in self.convs:
            x_meta = conv(x_meta, x_0_meta, meta_data.edge_index)
        x_meta = self.Linear4(x_meta)
        x_meta = x_meta.view(-1, 3, 8, 8)
        x_meta = self.starnet4(x_meta)

        x_GCNII = torch.cat((x_dis, x_meta), dim=0)

        ############
        tensor = self.att_embeddings_nope(processed_features)
        tensor_x = tensor
        # transformer encoder
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        x_former = self.final_ln(tensor)
        for dec_layer in self.layers2:
            tensor = dec_layer(tensor_x, x_former)
        x_former = self.final_ln2(tensor)

        target = x_former[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        split_tensor = torch.split(x_former, [1, self.seq_len - 1], dim=1)
        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        x_former = (node_tensor + neighbor_tensor).squeeze()


        ######
        output = torch.cat((x_GCNII, x_former), dim=1)
        x = output
        output = self.Linear2(output)
        output, vec_mean, vec_cov, = self.ib(output)
        output = F.softmax(output, dim=1)
        output = self.Linear3(output)  +  x

        embedings1 = self.mlp(output)
        embedings2 = self.mlp2(output)
        embedings = embedings1 * embedings2
        x1 = self.decoder(embedings)
        return x1, vec_mean, vec_cov





