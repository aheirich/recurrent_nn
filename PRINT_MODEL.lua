require 'torch'
require 'nn'

require 'LanguageModel'


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'trained/elman_shakespeare_35000.t7')
cmd:option('-length', 2000)
cmd:option('-start_text', '')
cmd:option('-sample', 1)
cmd:option('-temperature', 1)
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-verbose', 0)
local opt = cmd:parse(arg)

local checkpoint = torch.load(opt.checkpoint)
local p = checkpoint.model:parameters()
print(p)
print('p[0]',p[0])
print('p[1]',p[1])
print('p[2]',p[2])
print('p[3]',p[3])
print('p[4]',p[4])
print('p[5]',p[5])
print('p[6]',p[6])
print('p[7]',p[7])
print('p[8]',p[8])
print('p[9]',p[9])

