-- Copyright 2016 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

require 'image'
local TF = require 'transforms'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'stn'
require 'spy'
local flowX = require 'flowExtensions'

local M = {}

local eps = 1e-6
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

local mean = meanstd.mean
local std = meanstd.std
------------------------------------------
local function createWarpModel()
  local imgData = nn.Identity()()
  local floData = nn.Identity()()

  local imgOut = nn.Transpose({2,3},{3,4})(imgData)
  local floOut = nn.Transpose({2,3},{3,4})(floData)

  local warpImOut = nn.Transpose({3,4},{2,3})(nn.BilinearSamplerBHWD()({imgOut, floOut}))
  local model = nn.gModule({imgData, floData}, {warpImOut})

  return model
end

local down2 = nn.SpatialAveragePooling(2,2,2,2):cuda()
local down3 = nn.SpatialAveragePooling(2,2,2,2):cuda()
local down4 = nn.SpatialAveragePooling(2,2,2,2):cuda()
local down5 = nn.SpatialAveragePooling(2,2,2,2):cuda()
local down6 = nn.SpatialAveragePooling(2,2,2,2):cuda()
local down7 = nn.SpatialAveragePooling(2,2,2,2):cuda()
local down8 = nn.SpatialAveragePooling(2,2,2,2):cuda()
local down9 = nn.SpatialAveragePooling(2,2,2,2):cuda()


local warpmodel2 = createWarpModel():cuda()
local warpmodel3 = createWarpModel():cuda()
local warpmodel4 = createWarpModel():cuda()
local warpmodel5 = createWarpModel():cuda()
local warpmodel6 = createWarpModel():cuda()
local warpmodel7 = createWarpModel():cuda()
local warpmodel8 = createWarpModel():cuda()
local warpmodel9 = createWarpModel():cuda()

down2:evaluate()
down3:evaluate()
down4:evaluate()
down5:evaluate()
down6:evaluate()
down7:evaluate()
down8:evaluate()
down9:evaluate()


warpmodel2:evaluate()
warpmodel3:evaluate()
warpmodel4:evaluate()
warpmodel5:evaluate()
warpmodel6:evaluate()
warpmodel7:evaluate()
warpmodel8:evaluate()
warpmodel9:evaluate()

-------------------------------------------------
local  modelL0, modelL1, modelL2, modelL3, modelL4, modelL5, modelL6, modelL7, modelL8, modelL9

local modelL1path, modelL2path, modelL3path, modelL4path, modelL5path, modelL6path, modelL7path, modelL8path, modelL9path


modelL1path = paths.concat('models', 'modelG0.t7')
modelL2path = paths.concat('models', 'modelU0.t7')
modelL3path = paths.concat('models', 'modelG1.t7')
modelL4path = paths.concat('models', 'modelU1.t7')
modelL5path = paths.concat('models', 'modelG2.t7')
modelL6path = paths.concat('models', 'modelU2.t7')
modelL7path = paths.concat('models', 'modelG3.t7')
modelL8path = paths.concat('models', 'modelU3.t7')
modelL9path = paths.concat('models', 'modelG4.t7')

modelL1 = torch.load(modelL1path)
if torch.type(modelL1) == 'nn.DataParallelTable' then
   modelL1 = modelL1:get(1)
end
modelL1:evaluate()

modelL2 = torch.load(modelL2path)
if torch.type(modelL2) == 'nn.DataParallelTable' then
   modelL2 = modelL2:get(1)
end
modelL2:evaluate()

modelL3 = torch.load(modelL3path)
if torch.type(modelL3) == 'nn.DataParallelTable' then
   modelL3 = modelL3:get(1)
end
modelL3:evaluate()

modelL4 = torch.load(modelL4path)
if torch.type(modelL4) == 'nn.DataParallelTable' then
   modelL4 = modelL4:get(1)
end
modelL4:evaluate()

modelL5 = torch.load(modelL5path)
if torch.type(modelL5) == 'nn.DataParallelTable' then
   modelL5 = modelL5:get(1)
end
modelL5:evaluate()

modelL6 = torch.load(modelL6path)
if torch.type(modelL6) == 'nn.DataParallelTable' then
   modelL6 = modelL6:get(1)
end
modelL6:evaluate()

modelL7 = torch.load(modelL7path)
if torch.type(modelL7) == 'nn.DataParallelTable' then
   modelL7 = modelL7:get(1)
end
modelL7:evaluate()

modelL8 = torch.load(modelL8path)
if torch.type(modelL8) == 'nn.DataParallelTable' then
   modelL8 = modelL8:get(1)
end
modelL8:evaluate()

modelL9 = torch.load(modelL9path)
if torch.type(modelL9) == 'nn.DataParallelTable' then
   modelL9 = modelL9:get(1)
end
modelL9:evaluate()

local function getTrainValidationSplits(path)
   local numSamples = sys.fexecute( "ls " .. opt.data .. "| wc -l")/3
   local ff = torch.DiskFile(path, 'r')
   local trainValidationSamples = torch.IntTensor(numSamples)
   ff:readInt(trainValidationSamples:storage())
   ff:close()

   local train_samples = trainValidationSamples:eq(1):nonzero()
   local validation_samples = trainValidationSamples:eq(2):nonzero()

   return train_samples, validation_samples
  -- body
end
M.getTrainValidationSplits = getTrainValidationSplits

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   return input
end
M.loadImage = loadImage

local function loadFlow(filename)
  TAG_FLOAT = 202021.25 
  local ff = torch.DiskFile(filename):binary()
  local tag = ff:readFloat()
  if tag ~= TAG_FLOAT then
    xerror('unable to read '..filename..
     ' perhaps bigendian error','readflo()')
  end
   
  local w = ff:readInt()
  local h = ff:readInt()
  local nbands = 2
  local tf = torch.FloatTensor(h, w, nbands)
  ff:readFloat(tf:storage())
  ff:close()

  local flow = tf:permute(3,1,2)  
  return flow
end
M.loadFlow = loadFlow

local function computeInitFlowL1(imagesL1)
  local h = imagesL1:size(3)
  local w = imagesL1:size(4)
  local batchSize = imagesL1:size(1)

  local _flowappend = torch.zeros(batchSize, 2, h, w):cuda()
  local images_in = torch.cat(imagesL1, _flowappend, 2)

  local flow_est = modelL1:forward(images_in)
  return flow_est
end
M.computeInitFlowL1 = computeInitFlowL1

local function ScaleFlowL2(imagesL2)
  local imagesL1 = imagesL2:clone()
  local _flowappend = modelL2:forward(computeInitFlowL1(imagesL1))    
  return _flowappend
end
M.ScaleFlowL2 = ScaleFlowL2

local function computeInitFlowL3(imagesL3)
  local imagesL2 = down3:forward(imagesL3:clone()) 
  local _flowappend = ScaleFlowL2(imagesL2)     
  local _img3 = imagesL3[{{},{4,6},{},{}}]
  imagesL3[{{},{4,6},{},{}}]:copy(warpmodel3:forward({_img3, _flowappend}))

  local images_in = torch.cat(imagesL3, _flowappend, 2)
  
  local  flow_est = modelL3:forward(images_in)
  return flow_est:add(_flowappend)
end
M.computeInitFlowL3 = computeInitFlowL3

local function ScaleFlowL4(imagesL4)
  local imagesL3 = imagesL4:clone()
  local _flowappend = modelL4:forward(computeInitFlowL3(imagesL3))  
  return _flowappend
end
M.ScaleFlowL4 = ScaleFlowL4

local  function computeInitFlowL5(imagesL5)
  local imagesL4 = down5:forward(imagesL5)
  local _flowappend = ScaleFlowL4(imagesL4)  
  local _img5 = imagesL5[{{},{4,6},{},{}}]
  imagesL5[{{},{4,6},{},{}}]:copy(warpmodel5:forward({_img5, _flowappend}))

  local images_in = torch.cat(imagesL5, _flowappend, 2)
  
  local  flow_est = modelL5:forward(images_in)
  return flow_est:add(_flowappend)
end
M.computeInitFlowL5 = computeInitFlowL5

local function ScaleFlowL6(imagesL6)
  local imagesL5 = imagesL6:clone()
  local _flowappend = modelL6:forward(computeInitFlowL5(imagesL5))  
  return _flowappend
end
M.ScaleFlowL6 = ScaleFlowL6

local  function computeInitFlowL7(imagesL7)
  local imagesL6 = down7:forward(imagesL7)
  local _flowappend = ScaleFlowL6(imagesL6)  
  local _img7 = imagesL7[{{},{4,6},{},{}}]
  imagesL7[{{},{4,6},{},{}}]:copy(warpmodel7:forward({_img7, _flowappend}))

  local images_in = torch.cat(imagesL7, _flowappend, 2)
  
  local  flow_est = modelL7:forward(images_in)
  return flow_est:add(_flowappend)
end
M.computeInitFlowL7 = computeInitFlowL7

local function ScaleFlowL8(imagesL8)
  local imagesL7 = imagesL8:clone()
  local _flowappend = modelL8:forward(computeInitFlowL7(imagesL7)) 
  return _flowappend
end
M.ScaleFlowL8 = ScaleFlowL8

local  function computeInitFlowL9(imagesL9)
  local imagesL8 = down9:forward(imagesL9)
  local _flowappend = ScaleFlowL8(imagesL8)  
  local _img9 = imagesL9[{{},{4,6},{},{}}]
  imagesL9[{{},{4,6},{},{}}]:copy(warpmodel9:forward({_img9, _flowappend}))

  local images_in = torch.cat(imagesL9, _flowappend, 2)
  
  local  flow_est = modelL9:forward(images_in)
  return flow_est:add(_flowappend)
end
M.computeInitFlowL9 = computeInitFlowL9
local function getRawData(id)
   local path1 = paths.concat(opt.data, (string.format("%05i", id) .."_img1.ppm"))
   local path2 = paths.concat(opt.data, (string.format("%05i", id) .."_img2.ppm"))
   
   local img1 = loadImage(path1)
   local img2 = loadImage(path2)
   
   local pathF = paths.concat(opt.data, (string.format("%05i", id) .."_flow.flo"))
   local flow = loadFlow(pathF)

   return img1, img2, flow
end
M.getRawData = getRawData

local testHook = function(id)
   local path1 = paths.concat(opt.data, (string.format("%05i", id) .."_img1.ppm"))
   local path2 = paths.concat(opt.data, (string.format("%05i", id) .."_img2.ppm"))
   
   local img1 = loadImage(path1)
   local img2 = loadImage(path2)
   local images = torch.cat(img1, img2, 1)
   
   local pathF = paths.concat(opt.data, (string.format("%05i", id) .."_flow.flo"))
   local flow = loadFlow(pathF)
   
   images = TF.ColorNormalize(meanstd)(images)
   return images, flow
end
M.testHook = testHook

return M
