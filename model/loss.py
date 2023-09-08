import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, w, h, lamda, beta):
        super().__init__()
        self.w = w
        self.h = h
        self.lamda = lamda
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.l_lst = nn.Parameter(torch.zeros([]))

    def forward(self, img, present_res, next_res, gt):
        l_reg = self.mse_loss(present_res[:-1], gt[:-1]) * 0.5
        shape_h, shape_w = next_res.shape[2:]
        divide_points_h = [int(shape_h / self.h * i) for i in range(self.h)]
        divide_points_w = [int(shape_w / self.w * i) for i in range(self.w)]
        divide_points_h.append(shape_h)
        divide_points_w.append(shape_w)
        img_shape_h, img_shape_w = img.shape[2:]
        img_divide_points_h = [int(img_shape_h / self.h * i) for i in range(self.h)]
        img_divide_points_w = [int(img_shape_w / self.w * i) for i in range(self.w)]
        img_divide_points_h.append(img_shape_h)
        img_divide_points_w.append(img_shape_w)
        l_lst = 0
        for t in range(next_res.shape[0]-1):
            for r in range(self.h):
                for c in range(self.w):
                    it = img[t, :, img_divide_points_h[r]: img_divide_points_h[r+1],
                             img_divide_points_w[c]: img_divide_points_w[c+1]]
                    it_p1 = img[t+1, :, img_divide_points_h[r]: img_divide_points_h[r+1],
                                img_divide_points_w[c]: img_divide_points_w[c+1]]
                    m_lst_p1 = next_res[t, :, divide_points_h[r]: divide_points_h[r+1],
                                        divide_points_w[c]: divide_points_w[c+1]]
                    m_gt_p1 = gt[t+1, :, divide_points_h[r]: divide_points_h[r+1],
                                 divide_points_w[c]: divide_points_w[c+1]]
                    s = torch.exp(self.mse_loss(it, it_p1) * (-1) / 2 / (self.beta ** 2))
                    ms = self.mse_loss(m_lst_p1, m_gt_p1) / 2
                    l_lst += s * ms
        l_lst = l_lst / (next_res.shape[0] - 1)
        loss = l_reg + self.lamda * l_lst
        return loss


if __name__ == '__main__':
    device = torch.device('mps')
    criterion = Loss(2, 2, 0.001, 30.).to(device)
    a = torch.randn([5, 1, 64, 64]).to(device)
    b = torch.randn([5, 1, 64, 64]).to(device)
    c = torch.randn([5, 1, 64, 64]).to(device)
    d = torch.randn([5, 1, 64, 64]).to(device)
    e = criterion(a, b, c, d)
