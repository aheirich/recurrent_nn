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
for i = 1, #p do print('p[', i, '] = ', p[i]) end

