import os
import tempfile

import torch
import numpy as np
import mock

from tf_yarn.pytorch import model_ckpt


MODULE_UNDER_TEST = "tf_yarn.pytorch.model_ckpt"


def test_model_checkpointing():
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super(DummyModel, self).__init__()
            self.fc = torch.nn.Linear(13, 13)

        def forward(self) -> None:
            pass

    with tempfile.TemporaryDirectory() as tmp:
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        ckpt_path = model_ckpt.save_ckpt(tmp, model, optimizer, 0, **{"key": "value"})
        new_model = DummyModel()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.001, momentum=0.9)
        ckpt_dict = model_ckpt.load_ckpt(ckpt_path, new_model, new_optimizer, "cpu")

        new_model_params = list(new_model.parameters())
        model_params = list(model.parameters())
        assert len(new_model_params) == len(model_params)
        for n in range(len(new_model_params)):
            np.testing.assert_array_almost_equal(
                new_model_params[n].detach().numpy(),
                model_params[n].detach().numpy()
            )
        assert new_optimizer.state_dict() == optimizer.state_dict()
        assert ckpt_dict.get("key", None) == "value"
        assert ckpt_dict.get("epoch", None) == 0
