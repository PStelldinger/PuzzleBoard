import numpy as np
from scipy.signal import correlate2d

def _collapse_bits(code):
    res = np.zeros((3,code.shape[1]))
    ctr = res.copy()
    for i in range(0, code.shape[0]//3):
        res = res + code[3 * i : 3 * i + 3 , :]
        ctr = ctr + 1
    i = code.shape[0]//3
    res[0:code.shape[0]-3*i,:] = res[0:code.shape[0]-3*i,:] + code[3*i:,:]
    ctr[0:code.shape[0]-3*i,:] = ctr[0:code.shape[0]-3*i,:] + 1
    return res / ctr


class PuzzleBoard:

    code1 = np.array([[0,0,0,0,1,0,1,1,1,0,0,0,0,1,1,1,0,1,0,1,0,1,0,1,0,0,1,0,0,0,1,1,0,1,0,1,1,0,1,1,0,0,0,0,1,1,1,0,1,1,1,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,0,1,0,1,0,1,1,1,0,1,0,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,0,1,1,0,1,0,1,1,0,0,1,1,0,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,0,1], 
                      [0,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,1,0,1,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,0,0,1,1,0,0,0,1,1,1,0,1,0,1,0,0,0,1,1,1,0,0,0,1,0,0,1,1,0,0,0,1,1,1,1,0,1,0,1,0,0,1,0,0],
                      [0,1,0,0,0,0,0,1,0,1,0,0,0,0,1,1,1,0,1,0,0,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,1,0,1,0,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,1,1,0,0,1,1,0,0,1,0,1,1,1,1,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,1,1,0,1,1,1,1,1,0,1,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,1,1,1,0,1,0,0,0,1,1]])*2-1
    code2 = np.array([[0,0,1,1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,1,0,1,1,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,1,0,1,0,0,0,1,1,1,1,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1],
                      [1,0,1,1,0,0,1,1,0,0,0,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,1,1,0,0,0,0,1,1,0,1,0,1,0,0,0,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,1,0,0,0,0,1,1,1,1,1,0,1,0,1,0,1,1,0,1,1,0,1,1,0,0,0,1,1,0,0,1,1,1,1,0,0,1,0,0,1,0,0,1,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,0,0,1],
                      [1,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,1,1,1,1,1,1,0,0,0,1,1,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,0,1,1,1,0,0,1,0]])*2-1


    def __init__(self, root, sub_dot, img):
        self.root = root
        dim = root.dimensions
        sizex = dim[0]+dim[2]+1
        sizey = dim[1]+dim[3]+1
        self.size = np.array([sizey, sizex])
        self.valid = np.full([sizey, sizex], False)
        self.node_id = np.full([sizey, sizex], -1)
        self.sub_dot = np.zeros([sizey, sizex, 2])
        self.hvalid = np.full([sizey, sizex - 1], False)
        self.hsub_dot = np.zeros([sizey, sizex - 1, 2])
        self.hbits = np.zeros([sizey, sizex - 1])
        self.vvalid = np.full([sizey - 1, sizex], False)
        self.vsub_dot = np.zeros([sizey - 1, sizex, 2])
        self.vbits = np.zeros([sizey - 1, sizex])
        
        node = root
        while True:
            node.get_root()
            px = node.offset[0][0] + dim[2]
            py = node.offset[1][0] + dim[3]
            if not self.valid[py, px]:
                self.sub_dot[py, px, :] = sub_dot[node.id, :]
                self.valid[py, px] = True
                self.node_id[py,px] = node.id
            node = node.next
            if node == root:
                break
        self.hvalid = np.logical_and(self.valid[:,:-1], self.valid[:,1:])
        self.hsub_dot = (self.sub_dot[:,:-1,:] + self.sub_dot[:,1:,:]) / 2
        self.vvalid = np.logical_and(self.valid[:-1,:], self.valid[1:,:])
        self.vsub_dot = (self.sub_dot[:-1,:,:] + self.sub_dot[1:,:,:]) / 2
        
        hl=(self.sub_dot[:,:-1,:].reshape(-1,2)).astype(int)
        hr=(self.sub_dot[:,1:,:].reshape(-1,2)).astype(int)
        hc=(self.hsub_dot.reshape(-1,2)).astype(int)
        hbase = np.fromiter(( (img[hl[idx, 0], hl[idx, 1]] + img[hr[idx, 0], hr[idx, 1]])/2 for idx in range(len(hl))),float)
        hvals = np.fromiter(( img[hc[idx, 0], hc[idx, 1]] for idx in range(len(hc))),float)

        vl=(self.sub_dot[:-1,:,:].reshape(-1,2)).astype(int)
        vr=(self.sub_dot[1:,:,:].reshape(-1,2)).astype(int)
        vc=(self.vsub_dot.reshape(-1,2)).astype(int)
        vbase = np.fromiter(( (img[vl[idx, 0], vl[idx, 1]] + img[vr[idx, 0], vr[idx, 1]])/2 for idx in range(len(vl))),float)
        vvals = np.fromiter(( img[vc[idx, 0], vc[idx, 1]] for idx in range(len(vc))),float)

        self.hbits = ((hvals>hbase)*2-1).reshape(self.hvalid.shape)*(self.hvalid>0)
        self.vbits = ((vvals>vbase)*2-1).reshape(self.vvalid.shape)*(self.vvalid>0)
        hcode = _collapse_bits(np.rot90(self.hbits[::-1,::-1]))
        vcode = _collapse_bits(self.vbits[::-1,::-1])
        hcode2 = _collapse_bits(np.rot90(self.hbits[::-1,::-1]))[::-1,::-1]
        vcode2 = _collapse_bits(self.vbits)
        hcode3 = _collapse_bits(np.rot90(self.hbits))[::-1,::-1]
        vcode3 = _collapse_bits(self.vbits)[::-1,::-1]
        hcode4 = _collapse_bits(np.rot90(self.hbits))
        vcode4 = _collapse_bits(self.vbits[::-1,::-1])[::-1,::-1]

        corrA1 = correlate2d(PuzzleBoard.code1, vcode, mode='same', boundary='wrap')
        corrA2 = correlate2d(PuzzleBoard.code2, hcode3, mode='same', boundary='wrap')
        mxA=min(np.max(corrA1), np.max(corrA2))+0.01*max(np.max(corrA1), np.max(corrA2))
        mx=mxA
        offs1 = np.array([1, (vcode.shape[1]-1)//2])
        offs2 = np.array([1, (hcode3.shape[1]-1)//2])
        pos1=np.unravel_index(corrA1.argmax(), corrA1.shape)-offs1
        pos2=np.unravel_index(corrA2.argmax(), corrA2.shape)-offs2
        pos1[0]=pos1[0]%3
        pos2[0]=pos2[0]%3
        rot = 2

        corrB1 = correlate2d(PuzzleBoard.code1, hcode4, mode='same', boundary='wrap')
        corrB2 = correlate2d(PuzzleBoard.code2, vcode3, mode='same', boundary='wrap')
        mxB=min(np.max(corrB1), np.max(corrB2))+0.01*max(np.max(corrB1), np.max(corrB2))
        if(mxB>mx):
            mx=mxB
            offs1 = np.array([1, (hcode4.shape[1]-1)//2])
            offs2 = np.array([1, (vcode3.shape[1]-1)//2])
            pos1=np.unravel_index(corrB1.argmax(), corrB1.shape)-offs1
            pos2=np.unravel_index(corrB2.argmax(), corrB2.shape)-offs2
            pos1[0]=pos1[0]%3
            pos2[0]=pos2[0]%3
            rot = 1

        corrC1 = correlate2d(PuzzleBoard.code1, hcode, mode='same', boundary='wrap')
        corrC2 = correlate2d(PuzzleBoard.code2, vcode4, mode='same', boundary='wrap')
        mxC=min(np.max(corrC1), np.max(corrC2))+0.01*max(np.max(corrC1), np.max(corrC2))
        if(mxC>mx):
            mx=mxC
            offs1 = np.array([1, (hcode.shape[1]-1)//2])
            offs2 = np.array([1, (vcode4.shape[1]-1)//2])
            pos1=np.unravel_index(corrC1.argmax(), corrC1.shape)-offs1
            pos2=np.unravel_index(corrC2.argmax(), corrC2.shape)-offs2
            pos1[0]=pos1[0]%3
            pos2[0]=pos2[0]%3
            rot = 3

        corrD1 = correlate2d(PuzzleBoard.code1, vcode2, mode='same', boundary='wrap')
        corrD2 = correlate2d(PuzzleBoard.code2, hcode2, mode='same', boundary='wrap')
        mxD=min(np.max(corrD1), np.max(corrD2))+0.01*max(np.max(corrD1), np.max(corrD2))
        if(mxD>mx):
            mx=mxD
            offs1 = np.array([1, (vcode2.shape[1]-1)//2])
            offs2 = np.array([1, (hcode2.shape[1]-1)//2])
            pos1=np.unravel_index(corrD1.argmax(), corrD1.shape)-offs1
            pos2=np.unravel_index(corrD2.argmax(), corrD2.shape)-offs2
            pos1[0]=pos1[0]%3
            pos2[0]=pos2[0]%3
            rot = 0

        pos = np.array([pos1[1]+167*((pos1[1]+pos2[0])%3), pos2[1]+167*((pos2[1]-pos1[0])%3)])
        for i in range(rot):
            self.valid = np.rot90(self.valid)
            self.hvalid, self.vvalid = np.rot90(self.vvalid), np.rot90(self.hvalid)
            self.hbits, self.vbits = np.rot90(self.vbits), np.rot90(self.hbits)
            self.sub_dot = np.rot90(self.sub_dot)
            self.hsub_dot, self.vsub_dot = np.rot90(self.vsub_dot), np.rot90(self.hsub_dot)
            self.node_id = np.rot90(self.node_id)
        xs,ys=np.meshgrid(range(self.valid.shape[1]),range(self.valid.shape[0]))
        vfullCode = np.tile(PuzzleBoard.code1,(167,3))
        hfullCode = np.tile(np.rot90(PuzzleBoard.code2[::-1,::-1]),(3,167))
        
        self.hcorrect = hfullCode[pos[1]:pos[1]+self.hbits.shape[0],pos[0]:pos[0]+self.hbits.shape[1]]
        if self.hcorrect.shape == self.hbits.shape:
           self.hcorrect = (self.hcorrect * self.hbits == 1)
        else:
           self.hcorrect=1+0*self.hbits
        self.vcorrect = vfullCode[pos[1]:pos[1]+self.vbits.shape[0],pos[0]:pos[0]+self.vbits.shape[1]]
        if self.vcorrect.shape == self.vbits.shape:
           self.vcorrect = (self.vcorrect * self.vbits == 1)
        else:
           self.vcorrect=1+0*self.vbits
        
        self.positions=np.stack((ys,xs))+pos[[1,0]].reshape((-1,1,1))
