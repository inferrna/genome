import numpy as np
import copy

oldkernel = \
"""
\n
__kernel void runnet(__global float *_vs, __global float *_gms, __global float *vsr, __global float *res_g) {
    uint i, j, gid = get_global_id(0);
    __global float *gms = _gms + gid*nvarsg;
    __global float *vs = _vs;
    float rsi = 0.0, _res = 0.0;
    for(i=0; i<ninpt; i++){
        rsi = 0.0;
        for(j=0; j<nvarsd; j++){
            rsi += vs[j]*gms[j];
        }
        vs += nvarsd;
        _res += fabs(rsi-vsr[i]);
        //_res = gms[j-1];
    }
    res_g[gid] = _res;
}
\n
"""
def countcns(topology):
    a = np.array(topology, dtype=np.uint32)
    return int((a[:-1]*a[1:]).sum())
def topconns(topology):
    a = np.array(topology, dtype=np.uint32)
    return np.array(a[:-1]*a[1:], dtype=np.uint32)


def genkern2(samples, topology, cmpfunc):
    dm = 1
    counters = "ijkzhiuy"
    sums = "abcdef"
    lneurons = ["lnr", "dnr"]
    a = np.array(topology, dtype=np.uint32)
    neuronss = np.concatenate((np.array([0], dtype=np.uint32), a.cumsum(),))
    conns = (a[:-1]*a[1:]).astype(np.uint32)
    sconns = np.concatenate((np.array([0], dtype=np.uint32), conns.cumsum(),))
    print("conns is ", conns)
    print("sconns is ", sconns)
    print("neuronss is ", neuronss)
    currner = 0
    nextner = 0
    currcon = 0
    n = 0
    ss = []
    ses = []
    dcs = [0]+len(a)*[a[0]]
    for n in range(0, len(a)-1):
        s = []
        kname = "runnet"
        s.append("#define samples {0}".format(samples))
        s.append("__kernel void "+kname+"(__global float *_gconns, __global float *_"+\
                 lneurons[(n+1)%2]+", __global float *gtargets, __global float *results){")
        s.append(dm*"\t"+"uint gid = get_global_id(0);")
        s.append(dm*"\t"+"uint x = gid%samples;") #"Samples" loop
        s.append(dm*"\t"+"float dnr[{0}]".format(max(a[n:]))+";// = {"+", ".join(["0.0"])+"};")
        s.append(dm*"\t"+"float lnr[{0}]".format(max(a[n:]))+";// = {"+", ".join(["0.0"])+"};")
        s.append(dm*"\t"+"__global float *g{0} = _{0} + x*DC;".format(lneurons[(n+1)%2])) #Data Count
        s.append(dm*"\t"+"__global float *gconns;")    #Conns shift and Conns number
        s.append(dm*"\t"+"float result = 0.0;")
        #s.append((dm)*"\t"+"__global float gneurons = _gneurons+{0};".format(neuronss[n])) #"Samples" loop
        #dm+=1
        currcon += a[n]*a[n+1]
        nextner += a[n]
        s.append(dm*"\t"+"gconns = _gconns+CS+CN*(gid/samples);")    #Conns shift and Conns number
        s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n]))+"{")
        s.append((dm+1)*"\t"+"{0}[{1}] = {2}{0}[{1}];".format(lneurons[(n+1)%2], counters[dm], ['g', 'g'][int(n>0)])) #was '_', 'g'
        s.append((dm+1)*"\t"+"{0}[{1}] = 0.0;".format(lneurons[n%2], counters[dm]))
        s.append(dm*"\t"+"}")
        for m in range(n, len(a)-1):
            s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[m]))+"{ //\"Neurons\" loop "+str(m)) #"Neurons" loop
            if(a[n+1]>1):
                s.append((dm+1)*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)"\
                                     .format(counters[dm+1], str(a[m+1]))\
                                     +"{ //\"Connections\" loop "+str(m)) #"Connections" loop
                nrcnt = counters[dm+1]
            else: nrcnt = 0
            s.append((dm+2)*"\t"+"{5}[{1}] += {6}[{3}]*gconns[{3}*{4}+{1}]".format(\
                                                                                           nextner, nrcnt,\
                                                                                           currner, counters[dm],\
                                                                                           str(a[m+1]), lneurons[m%2],\
                                                                                           lneurons[(m+1)%2])+";")
            if(a[n+1]>1):
                s.append((dm+1)*"\t"+"}") #"Connections" loop
            s.append((dm+1)*"\t"+"{0}[{1}] = 0.0;".format(lneurons[(m+1)%2], counters[dm]))
            s.append(dm*"\t"+"}") #"Neurons" loop
            s.append(dm*"\t"+"gconns += {0};".format(a[m]*a[m+1])) #"Neurons" loop
            if m==n:
                se = copy.copy(s)
                se.append((dm)*"\t"+"barrier(CLK_GLOBAL_MEM_FENCE);")
                se.append((dm)*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n+1])))
                se.append((dm+1)*"\t"+"g{0}[{1}] = {2}[{1}];".format(lneurons[(m+1)%2], counters[dm], lneurons[(m)%2]))
                se.append(dm*"\t"+"g{0} += {1};".format(lneurons[(n+1)%2], a[0]))
                se.append("}") #"Kernel" end
        currcon += a[n]*a[n+1]
        currner += a[n]
        #s.append(dm*"\t"+"g{0} += {1};".format(lneurons[(n+1)%2], a[0]))
        s.append((dm)*"\t"+"results[gid] = fabs(gtargets[{0}] - {1}[0]);".format("x", lneurons[m%2])) #"Samples" loop
        #s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n])))
        #s.append((dm+1)*"\t"+"{0}[{1}] = 0.0;".format(lneurons[m%2], counters[dm])) #"Samples" loop
        #s.append((dm)*"\t"+"results[gid] = result/samples;")
        s.append((dm-1)*"\t"+"}") #Kernel end
        s = ["#define SC {0}".format(samples), "#define DC {1} //Step for prev layer data by each gen".format(dcs[n], a[0]), "#define CS {0}".format(sconns[n]), "#define CN {0}".format(conns.sum())] + s
        se = ["#define SC 1", "#define DC {0} //Step for input data".format(a[0]), "#define CS {0}".format(sconns[n]), "#define CN 0"]+se
        ss.append(cmpfunc("\n".join(s)))
        ses.append(cmpfunc("\n".join(se)))
        if n==0:#len(a)-2:
            print("\n".join(s))
            print("\n".join(se))
    return {"ordinal":ss, "finish":ses}

