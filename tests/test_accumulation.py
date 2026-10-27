import copy,unittest
import torch
from trex_force.engine import accumulated_backward

class AccumulationTests(unittest.TestCase):
    def test_matches_replica_gradients_loss_and_primary_bn(self):
        torch.manual_seed(13)
        model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.BatchNorm1d(8),torch.nn.ReLU(),torch.nn.Linear(8,3))
        replica1,replica2=copy.deepcopy(model),copy.deepcopy(model)
        x,y=torch.randn(10,4),torch.randn(10,3)
        loss_fn=torch.nn.SmoothL1Loss()
        a,b=replica1(x[:5]),replica2(x[5:])
        loss1,loss2=loss_fn(a,y[:5])*.5,loss_fn(b,y[5:])*.5
        loss1.backward();loss2.backward()
        pred,loss=accumulated_backward(model,x,y,torch.zeros(3),torch.ones(3),loss_fn,
            torch.amp.GradScaler('cuda',enabled=False),torch.device('cpu'),None,5)
        torch.testing.assert_close(loss,loss1+loss2)
        torch.testing.assert_close(pred,torch.cat([a,b]))
        for p,a,b in zip(model.parameters(),replica1.parameters(),replica2.parameters()):
            torch.testing.assert_close(p.grad,a.grad+b.grad)
        for a,b in zip(model.buffers(),replica1.buffers()):torch.testing.assert_close(a,b)
