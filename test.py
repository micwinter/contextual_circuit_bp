#Compute input size that leads to a 1x1 output size, among other things   

# [filter size, stride, padding]

convnet =[[3,1,1],[2,1,1],[3,1,1],[2,1,1]]
layer_name = ['conv1','pool1','conv2','pool2']
imsize = 28

def outFromIn(isz, layernum = 9, net = convnet):
    if layernum>len(net): layernum=len(net)

    totstride = 1
    insize = isz
    #for layerparams in net:
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2*pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride

def inFromOut( layernum = 9, net = convnet):
    if layernum>len(net): layernum=len(net)
    outsize = 1
    #for layerparams in net:
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        outsize = ((outsize -1)* stride) + fsize
    RFsize = outsize
    return RFsize

if __name__ == '__main__':

    print "layer output sizes given image = %dx%d" % (imsize, imsize)
    for i in range(len(convnet)):
        p = outFromIn(imsize,i+1)
        print p
        rf = inFromOut(i+1)
        print "Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (layer_name[i], p[0], p[1], rf)
