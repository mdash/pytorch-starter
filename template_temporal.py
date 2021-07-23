from torch import nn
import torch
from typing import Dict, List, Tuple
from pytorch_forecasting.models.nn import MultiEmbedding
from pytorch_forecasting.models.base_model import BaseModelWithCovariates


class DilatedNet(nn.Module):
    def __init__(self, num_securities=5, hidden_size=64, dilation=2, T=10):
        """

        :param num_securities: int, number of stocks
        :param hidden_size: int, size of hidden layers
        :param dilation: int, dilation value
        :param T: int, number of look back points
        """
        super(DilatedNet, self).__init__()
        self.dilation = dilation
        self.hidden_size = hidden_size
        # First Layer
        # Input
        self.dilated_conv1 = nn.Conv1d(
            num_securities, hidden_size, kernel_size=2, dilation=self.dilation)
        self.relu1 = nn.ReLU()

        # Layer 2
        self.dilated_conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu2 = nn.ReLU()

        # Layer 3
        self.dilated_conv3 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu3 = nn.ReLU()

        # Layer 4
        self.dilated_conv4 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu4 = nn.ReLU()

        # Output layer
        self.conv_final = nn.Conv1d(hidden_size, num_securities, kernel_size=1)

        self.T = T

    def forward(self, x):
        """

        :param x: Pytorch Variable, batch_size x n_stocks x T
        :return:
        """

        # First layer
        out = self.dilated_conv1(x)
        out = self.relu1(out)

        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)

        # Layer 3:
        out = self.dilated_conv3(out)
        out = self.relu3(out)

        # Layer 4:
        out = self.dilated_conv4(out)
        out = self.relu4(out)

        # Final layer
        out = self.conv_final(out)
        out = out[:, :, -1]

        return out


class FullyConnectedModelWithCovariates(BaseModelWithCovariates):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        x_reals: List[str],
        x_categoricals: List[str],
        embedding_sizes: Dict[str, Tuple[int, int]],
        embedding_labels: Dict[str, List[str]],
        static_categoricals: List[str],
        static_reals: List[str],
        time_varying_categoricals_encoder: List[str],
        time_varying_categoricals_decoder: List[str],
        time_varying_reals_encoder: List[str],
        time_varying_reals_decoder: List[str],
        embedding_paddings: List[str],
        categorical_groups: Dict[str, List[str]],
        **kwargs,
    ):
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)

        # create embedder - can be fed with x["encoder_cat"] or x["decoder_cat"] and will return
        # dictionary of category names mapped to embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )

        # calculate the size of all concatenated embeddings + continous variables
        n_features = sum(
            embedding_size for classes_size, embedding_size in self.hparams.embedding_sizes.values()
        ) + len(self.reals)

        # create network that will be fed with continious variables and embeddings
        self.network = FullyConnectedModule(
            input_size=self.hparams.input_size * n_features,
            output_size=self.hparams.output_size,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        batch_size = x["encoder_lengths"].size(0)
        # returns dictionary with embedding tensors
        embeddings = self.input_embeddings(x["encoder_cat"])
        network_input = torch.cat(
            [x["encoder_cont"]]
            + [
                emb
                for name, emb in embeddings.items()
                if name in self.encoder_variables or name in self.static_variables
            ],
            dim=-1,
        )
        prediction = self.network(network_input.view(batch_size, -1))

        # rescale predictions into target space
        prediction = self.transform_output(
            prediction, target_scale=x["target_scale"])

        # We need to return a dictionary that at least contains the prediction.
        # The parameter can be directly forwarded from the input.
        # The conversion to a named tuple can be directly achieved with the `to_network_output` function.
        return self.to_network_output(prediction=prediction)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = {
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }
        # use to pass real hyperparameters and override defaults set by dataset
        new_kwargs.update(kwargs)
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"

        return super().from_dataset(dataset, **new_kwargs)
