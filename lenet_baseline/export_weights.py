import torch, struct, numpy as np, argparse

parser=argparse.ArgumentParser()
parser.add_argument('--pth',default='lenet5_std.pth')
parser.add_argument('--out',default='weight/lenet5_fp32.bin')
args=parser.parse_args()

state=torch.load(args.pth,map_location='cpu')
order=['conv1.weight','conv1.bias',
       'conv2.weight','conv2.bias',
       'fc1.weight','fc1.bias',
       'fc2.weight','fc2.bias']
with open(args.out,'wb') as f:
    for k in order:
        arr=state[k].view(-1).cpu().numpy().astype(np.float32)
        f.write(arr.tobytes())
print('Wrote',args.out)

