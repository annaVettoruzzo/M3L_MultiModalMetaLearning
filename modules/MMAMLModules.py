from collections import OrderedDict
import torch
import numpy as np
import torch.nn.functional as F
from utils import weight_init

# -------------------------------------------------------------------
class GatedNet(torch.nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_sizes=[40, 40],
                 nonlinearity=F.relu, condition_type='affine', condition_order='low2high'):
        super(GatedNet, self).__init__()
        self._nonlinearity = nonlinearity
        self._condition_type = condition_type
        self._condition_order = condition_order

        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers):
            self.add_module(
                'layer{0}_linear'.format(i),
                torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.add_module(
            'output_linear',
            torch.nn.Linear(layer_sizes[self.num_layers - 1],
                            layer_sizes[self.num_layers]))
        self.apply(weight_init)

    def conditional_layer(self, x, embedding):
        if self._condition_type == 'sigmoid_gate':
            x = x * F.sigmoid(embedding).expand_as(x)
        elif self._condition_type == 'affine':
            gammas, betas = torch.split(embedding, x.size(1), dim=-1)
            gammas = gammas + torch.ones_like(gammas)
            x = x * gammas + betas
        elif self._condition_type == 'softmax':
            x = x * F.softmax(embedding).expand_as(x)
        else:
            raise ValueError('Unrecognized conditional layer type {}'.format(
                self._condition_type))
        return x

    def forward(self, x, params=None, embeddings=None, training=True):
        if params is None:
            params = OrderedDict(self.named_parameters())

        if embeddings is not None:
            if self._condition_order == 'high2low':  # High2Low
                embeddings = {'layer{}_linear'.format(len(params) - i): embedding
                              for i, embedding in enumerate(embeddings[::-1])}
            elif self._condition_order == 'low2high':  # Low2High
                embeddings = {'layer{}_linear'.format(i): embedding
                              for i, embedding in enumerate(embeddings[::-1], start=1)}
            else:
                raise NotImplementedError(
                    'Unsuppported order for using conditional layers')
        # x = task.x.view(task.x.size(0), -1)

        for key, module in self.named_modules():
            if 'linear' in key:
                x = F.linear(x, weight=params[key + '.weight'],
                             bias=params[key + '.bias'])
                if 'output' not in key and embeddings is not None:  # conditioning and nonlinearity
                    if type(embeddings.get(key, -1)) != type(-1):
                        x = self.conditional_layer(x, embeddings[key])

                    x = self._nonlinearity(x)

        return x


class GatedConvModel(torch.nn.Module):
    def __init__(self, input_channels=3, output_size=5, num_channels=64,
                 kernel_size=3, padding=1, nonlinearity=F.relu,
                 use_max_pool=False, img_side_len=32,
                 condition_type='affine', condition_order='low2high',
                 kml=False, verbose=False):
        super(GatedConvModel, self).__init__()
        self._input_channels = input_channels
        self._output_size = output_size
        self._num_channels = num_channels
        self._kernel_size = kernel_size
        self._nonlinearity = nonlinearity
        self._use_max_pool = use_max_pool
        self._padding = padding
        self._condition_type = condition_type
        self._condition_order = condition_order
        self._bn_affine = False
        self._reuse = False
        self._verbose = verbose
        self.kml = kml

        if self._use_max_pool:
            self._conv_stride = 1
            self._features_size = 1
            self.features = torch.nn.Sequential(OrderedDict([
                ('layer1_conv', torch.nn.Conv2d(self._input_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer1_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer1_condition', None),
                ('layer1_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                       stride=2)),
                ('layer1_relu', torch.nn.ReLU(inplace=True)),
                ('layer2_conv', torch.nn.Conv2d(self._num_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer2_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer2_condition', None),
                ('layer2_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                       stride=2)),
                ('layer2_relu', torch.nn.ReLU(inplace=True)),
                ('layer3_conv', torch.nn.Conv2d(self._num_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer3_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer3_condition', None),
                ('layer3_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                       stride=2)),
                ('layer3_relu', torch.nn.ReLU(inplace=True)),
                ('layer4_conv', torch.nn.Conv2d(self._num_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer4_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer4_condition', None),
                ('layer4_max_pool', torch.nn.MaxPool2d(kernel_size=2,
                                                       stride=2)),
                ('layer4_relu', torch.nn.ReLU(inplace=True)),
            ]))
        else:
            self._conv_stride = 2
            self._features_size = (img_side_len // 14) ** 2
            self.features = torch.nn.Sequential(OrderedDict([
                ('layer1_conv', torch.nn.Conv2d(self._input_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer1_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer1_condition', torch.nn.ReLU(inplace=True)),
                ('layer1_relu', torch.nn.ReLU(inplace=True)),
                ('layer2_conv', torch.nn.Conv2d(self._num_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer2_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer2_condition', torch.nn.ReLU(inplace=True)),
                ('layer2_relu', torch.nn.ReLU(inplace=True)),
                ('layer3_conv', torch.nn.Conv2d(self._num_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer3_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer3_condition', torch.nn.ReLU(inplace=True)),
                ('layer3_relu', torch.nn.ReLU(inplace=True)),
                ('layer4_conv', torch.nn.Conv2d(self._num_channels,
                                                self._num_channels,
                                                self._kernel_size,
                                                stride=self._conv_stride,
                                                padding=self._padding)),
                ('layer4_bn', torch.nn.BatchNorm2d(self._num_channels,
                                                   affine=self._bn_affine,
                                                   momentum=0.001)),
                ('layer4_condition', torch.nn.ReLU(inplace=True)),
                ('layer4_relu', torch.nn.ReLU(inplace=True)),
            ]))
        self.classifier = torch.nn.Sequential(OrderedDict([
            ('fully_connected', torch.nn.Linear(self._num_channels,
                                                self._output_size))
        ]))
        self.apply(weight_init)

    def conditional_layer(self, x, embedding):
        if self._condition_type == 'sigmoid_gate':
            x = x * F.sigmoid(embedding).expand_as(x)
        elif self._condition_type == 'affine':
            gammas, betas = torch.split(embedding, x.size(1), dim=-1)
            gammas = gammas.view(1, -1, 1, 1).expand_as(x)
            betas = betas.view(1, -1, 1, 1).expand_as(x)
            gammas = gammas + torch.ones_like(gammas)
            x = x * gammas + betas
        elif self._condition_type == 'softmax':
            x = x * F.softmax(embedding).view(1, -1, 1, 1).expand_as(x)
        else:
            raise ValueError('Unrecognized conditional layer type {}'.format(
                self._condition_type))
        return x

    def forward(self, x, params=None, embeddings=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        if self.kml==False: #standard MMAML
            if embeddings is not None:
                embeddings = {'layer{}_condition'.format(i): embedding
                                for i, embedding in enumerate(embeddings, start=1)}

            for layer_name, layer in self.features.named_children():
                weight = params.get('features.' + layer_name + '.weight', None)
                bias = params.get('features.' + layer_name + '.bias', None)
                if 'conv' in layer_name:
                    x = F.conv2d(x, weight=weight, bias=bias,
                                 stride=self._conv_stride, padding=self._padding)
                elif 'condition' in layer_name:
                    x = self.conditional_layer(x, embeddings[layer_name]) if embeddings is not None else x
                elif 'bn' in layer_name:
                    x = F.batch_norm(x, weight=weight, bias=bias,
                                     running_mean=layer.running_mean,
                                     running_var=layer.running_var,
                                     training=True)
                elif 'max_pool' in layer_name:
                    x = F.max_pool2d(x, kernel_size=2, stride=2)
                elif 'relu' in layer_name:
                    x = F.relu(x)
                elif 'fully_connected' in layer_name:
                    break
                else:
                    raise ValueError('Unrecognized layer {}'.format(layer_name))
                if not self._reuse and self._verbose: print('{}: {}'.format(layer_name, x.size()))

            # in maml network the conv maps are average pooled
            x = x.view(x.size(0), self._num_channels, self._features_size)
            if not self._reuse and self._verbose: print('reshape to: {}'.format(x.size()))
            x = torch.mean(x, dim=2)
            if not self._reuse and self._verbose: print('reduce mean: {}'.format(x.size()))
            logits = F.linear(
                x, weight=params['classifier.fully_connected.weight'],
                bias=params['classifier.fully_connected.bias'])
            if not self._reuse and self._verbose: print('logits size: {}'.format(logits.size()))
            if not self._reuse and self._verbose: print('='*27)
            self._reuse = True

            return logits

        else: #MMAML with KML (revisited version)
            cnt_embedding = 0
            for layer_name, layer in self.features.named_children():
                weight = params.get('features.' + layer_name + '.weight', None)
                bias = params.get('features.' + layer_name + '.bias', None)
                if 'conv' in layer_name:
                    # Extract kernel modulation parameters and apply modulation
                    v_mod = embeddings[cnt_embedding]
                    u_mod = embeddings[cnt_embedding + 1]
                    w_mod = torch.ger(v_mod.squeeze(), u_mod.squeeze())
                    b_mod = embeddings[cnt_embedding + 2]

                    w_mod = w_mod.view(weight.shape)
                    b_mod = b_mod.view(bias.shape)

                    weight = weight * w_mod
                    bias = bias + b_mod
                    cnt_embedding += 3
                    # Perform inference using the modulated parameters
                    x = F.conv2d(x, weight=weight, bias=bias,
                                 stride=self._conv_stride, padding=self._padding)
                elif 'condition' in layer_name:
                    pass
                elif 'bn' in layer_name:
                    x = F.batch_norm(x, weight=weight, bias=bias,
                                     running_mean=layer.running_mean,
                                     running_var=layer.running_var,
                                     training=True)
                elif 'max_pool' in layer_name:
                    x = F.max_pool2d(x, kernel_size=2, stride=2)
                elif 'relu' in layer_name:
                    x = F.relu(x)
                elif 'fully_connected' in layer_name:
                    break
                else:
                    raise ValueError('Unrecognized layer {}'.format(layer_name))
                if not self._reuse and self._verbose: print('{}: {}'.format(layer_name, x.size()))

            x = x.view(x.size(0), self._num_channels, self._features_size)
            if not self._reuse and self._verbose: print('reshape to: {}'.format(x.size()))
            x = torch.mean(x, dim=2)
            if not self._reuse and self._verbose: print('reduce mean: {}'.format(x.size()))
            logits = F.linear(
                x, weight=params['classifier.fully_connected.weight'],
                bias=params['classifier.fully_connected.bias'])
            self._reuse = True

            return logits


# -------------------------------------------------------------------
class LSTMEmbeddingModel(torch.nn.Module):
    def __init__(self, input_size=1, output_size=1, embedding_dims=[80, 80], hidden_size=40, num_layers=2, device="cuda"):
        super(LSTMEmbeddingModel, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._embedding_dims = embedding_dims
        self._bidirectional = True
        self._device = device

        rnn_input_size = int(input_size + output_size)
        self.rnn = torch.nn.LSTM(
            rnn_input_size, hidden_size, num_layers, bidirectional=self._bidirectional)

        self._embeddings = torch.nn.ModuleList()
        for dim in embedding_dims:
            self._embeddings.append(torch.nn.Linear(
                hidden_size*(2 if self._bidirectional else 1), dim))

    def forward(self, tsk, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        batch_size = 1
        x, y = tsk.X_sp, tsk.y_sp
        h0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                         batch_size, self._hidden_size, device=self._device)
        c0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                         batch_size, self._hidden_size, device=self._device)
        # LSTM input dimensions are seq_len, batch, input_size
        inputs = torch.cat((x, y), dim=1).view(x.size(0), 1, -1)
        output, (hn, cn) = self.rnn(inputs, (h0, c0))
        if self._bidirectional:
            N, B, H = output.shape
            output = output.view(N, B, 2, H // 2)
            embedding_input = torch.cat(
                [output[-1, :, 0], output[0, :, 1]], dim=1)

        out_embeddings = []
        for embedding in self._embeddings:
            out_embeddings.append(embedding(embedding_input))
        return out_embeddings


class ConvEmbeddingModel(torch.nn.Module):
    def __init__(self, input_size=np.prod((3, 32, 32)), output_size=5, embedding_dims=[128, 128, 128, 128],
                 hidden_size=256, num_layers=2,
                 convolutional=True, num_conv=4, num_channels=32, num_channels_max=256,
                 rnn_aggregation=False, linear_before_rnn=False,
                 embedding_pooling='avg', batch_norm=True, avgpool_after_conv=True,
                 img_size=(3,32,32), verbose=False):

        super(ConvEmbeddingModel, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._embedding_dims = embedding_dims
        self._bidirectional = True
        self._device = "cuda"
        self._convolutional = convolutional
        self._num_conv = num_conv
        self._num_channels = num_channels
        self._num_channels_max = num_channels_max
        self._batch_norm = batch_norm
        self._img_size = img_size
        self._rnn_aggregation = rnn_aggregation
        self._embedding_pooling = embedding_pooling
        self._linear_before_rnn = linear_before_rnn
        self._embeddings_array = []
        self._avgpool_after_conv = avgpool_after_conv
        self._reuse = False
        self._verbose = verbose

        if self._convolutional:
            conv_list = OrderedDict([])
            num_ch = [self._img_size[0]] + [self._num_channels for i in range(self._num_conv)]
            num_ch = [min(num_channels_max, ch) for ch in num_ch]

            for i in range(self._num_conv):
                conv_list.update({
                    'conv{}'.format(i+1):
                        torch.nn.Conv2d(num_ch[i], num_ch[i+1],
                                        (3, 3), stride=2, padding=1)})
                if self._batch_norm:
                    conv_list.update({
                        'bn{}'.format(i+1):
                            torch.nn.BatchNorm2d(num_ch[i+1], momentum=0.001)})
                conv_list.update({'relu{}'.format(i+1): torch.nn.ReLU(inplace=True)})
            self.conv = torch.nn.Sequential(conv_list)
            self._num_layer_per_conv = len(conv_list) // self._num_conv

            if self._avgpool_after_conv:
                rnn_input_size = self.conv[self._num_layer_per_conv * (self._num_conv-1)].out_channels
            else:
                rnn_input_size = self.compute_input_size(
                    1, 3, 2, self.conv[self._num_layer_per_conv*(self._num_conv-1)].out_channels)
        else:
            rnn_input_size = int(input_size)

        self.rnn = None
        embedding_input_size = hidden_size
        self.linear = torch.nn.Linear(rnn_input_size, embedding_input_size)
        self.relu_after_linear = torch.nn.ReLU(inplace=True)

        self._embeddings = torch.nn.ModuleList()
        for dim in embedding_dims:
            self._embeddings.append(torch.nn.Linear(embedding_input_size, dim))

    def forward(self, tsk, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x, y = tsk.X_sp, tsk.y_sp
        for layer_name, layer in self.conv.named_children():
            weight = params.get('conv.' + layer_name + '.weight', None)
            bias = params.get('conv.' + layer_name + '.bias', None)
            if 'conv' in layer_name:
                x = F.conv2d(x, weight=weight, bias=bias, stride=2, padding=1)
            elif 'relu' in layer_name:
                x = F.relu(x)
            elif 'bn' in layer_name:
                x = F.batch_norm(x, weight=weight, bias=bias,
                                 running_mean=layer.running_mean,
                                 running_var=layer.running_var,
                                 training=True)
        if self._avgpool_after_conv:
            x = x.view(x.size(0), x.size(1), -1)
            x = torch.mean(x, dim=2)
        else:
            x = x.view(x.size(0), -1)

        if self._rnn_aggregation:
            # LSTM input dimensions are seq_len, batch, input_size
            batch_size = 1
            h0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                             batch_size, self._hidden_size, device=self._device)
            if self._linear_before_rnn:
                x = F.relu(self.linear(x))
            inputs = x.view(x.size(0), 1, -1)
            output, hn = self.rnn(inputs, h0)
            if self._bidirectional:
                N, B, H = output.shape
                output = output.view(N, B, 2, H // 2)
                embedding_input = torch.cat([output[-1, :, 0], output[0, :, 1]], dim=1)
        else:
            inputs = F.relu(self.linear(x).view(1, x.size(0), -1).transpose(1, 2))
            if not self._reuse and self._verbose: print('fc: {}'.format(inputs.size()))
            if self._embedding_pooling == 'avg':
                embedding_input = F.avg_pool1d(inputs, x.size(0)).view(1, -1)
            else:
                raise NotImplementedError

        out_embeddings = []
        for i, embedding in enumerate(self._embeddings):
            embedding_vec = embedding(embedding_input)
            out_embeddings.append(embedding_vec)
            if not self._reuse and self._verbose: print('emb vec {} size: {}'.format(
                i+1, embedding_vec.size()))
        if not self._reuse and self._verbose: print('='*27)
        self._reuse = True

        return out_embeddings

