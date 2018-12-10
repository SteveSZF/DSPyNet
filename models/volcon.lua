-- Copyright 2016 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
function createModel(nGPU)
   input = - nn.Identity()
   img = input - nn.SpatialConvolution(8,32,3,3,1,1,1,1) - nn.ReLU(true)
               - nn.SpatialConvolution(32,64,3,3,1,1,1,1) - nn.ReLU(true)
			   - nn.SpatialConvolution(64,96,3,3,1,1,1,1) - nn.ReLU(true)
   b1_1 = img - nn.SpatialConvolution(96,36,1,1,1,1,0,0) - nn.ReLU(true)
   b2_1 = img - nn.SpatialConvolution(96,48,1,1,1,1,0,0) - nn.ReLU(true)
			  - nn.SpatialConvolution(48,64,3,3,1,1,1,1) - nn.ReLU(true)
   b3_1 = img - nn.SpatialConvolution(96,16,1,1,1,1,0,0) - nn.ReLU(true)
		      - nn.SpatialConvolution(16,32,3,3,1,1,1,1) - nn.ReLU(true)
	          - nn.SpatialConvolution(32,48,3,3,1,1,1,1) - nn.ReLU(true)
   b4_1 = img - nn.SpatialMaxPooling(3,3,1,1,1,1) - nn.ReLU(true)
			  - nn.SpatialConvolution(96,12,1,1,1,1,0,0) - nn.ReLU(true)

   join1 = {b1_1,b2_1,b3_1,b4_1} - nn.JoinTable(2)
   local conv = nn.SpatialConvolution(64,64,3,3,1,1,1,1)																	  
   join3 = join1 - nn.SpatialConvolution(160,64,1,1,1,1,0,0) - nn.ReLU(true)
			     - nn.SpatialConvolution(64,64,3,3,1,1,1,1) - nn.ReLU(true)
			     - nn.SpatialConvolution(64,64,3,3,1,1,1,1) - nn.ReLU(true)
			     - nn.SpatialConvolution(64,64,3,3,1,1,1,1) - nn.ReLU(true)

   t = join3 - nn.SpatialConvolution(64,32,3,3,1,1,1,1) - nn.ReLU(true)
             - nn.SpatialConvolution(32,16,3,3,1,1,1,1) - nn.ReLU(true)
             - nn.SpatialConvolution(16,8,3,3,1,1,1,1) - nn.ReLU(true)
   r = t - nn.SpatialConvolution(8,2,7,7,1,1,3,3)
   model = nn.gModule({input},{r})

   if nGPU>0 then
      model:cuda()
      model = makeDataParallel(model, nGPU)
   end
   
   return model
end
