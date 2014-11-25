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
def genkern(samples, topology):
    dm = 1
    counters = "ijkzhiuy"
    sums = "abcdef"
    lneurons = ["lnr", "dnr"]
    a = np.array(topology, dtype=np.uint32)
    conns = int((a[:-1]*a[1:]).sum())
#s.append("for({0}=0; {0}<{1}; {0}++)".format(counters[dm], str(len(a)))+"{ //\"Layers\" loop")        #Layers loop
    currner = 0
    nextner = 0
    currcon = 0
    n = 0
#nextner += a[n]
#s.append(dm*"\t"+"sums[{1}][{0}]+=xxx;".format(counters[dm], dm-1))
    s = []
    s.append("__kernel void runnet(__global float *gneurons, __global float *ggconns, __global float *gtargets, __global float *results){")
    s.append(dm*"\t"+"float dnr[{0}];".format(a[1]))#+"{"+", ".join(a[1]*["0.0"])+"};")
    s.append(dm*"\t"+"float lnr[{0}];".format(a[1]))#+"{"+", ".join(a[1]*["0.0"])+"};")
    s.append(dm*"\t"+"float result = 0.0;")
    s.append(dm*"\t"+"uint totconns = {0};".format(conns))
    s.append(dm*"\t"+"uint gid = get_global_id(0);")
    s.append(dm*"\t"+"__global float *gconns = ggconns+{0}*gid;".format(conns))
    s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}+=1)".format("n_sam", samples)+"{ //\"Samples\" loop ") #"Samples" loop
    dm+=1
    s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n]))+"{ //\"Neurons\" loop "+str(n)) #"Neurons" loop
    s.append((dm+1)*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm+1], str(a[n+1]))+"{ //\"Connections\" loop "+str(n)) #"Connections" loop
    s.append((dm+2)*"\t"+"{6}[{1}] += gneurons[{3}]*gconns[{3}*{5}+{1}]".format(\
                                                                                   nextner, counters[dm+1],\
                                                                                   "n_sam", counters[dm],\
                                                                                   currcon, str(a[n+1]), 
                                                                                   lneurons[n%2])+";\n"+(dm+1)*"\t"+"}") #"Connections" loop
    s.append(dm*"\t"+"}") #"Connections" loop
    currcon += a[n]*a[n+1]
    for n in range(1, len(a)-1):
        nextner += a[n]
        #s.append(dm*"\t"+"sums[{1}][{0}]+=xxx;".format(counters[dm], dm-1))
        s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n]))+"{ //\"Neurons\" loop "+str(n)) #"Neurons" loop
        if(a[n+1]>1):
            s.append((dm+1)*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)"\
                                 .format(counters[dm+1], str(a[n+1]))\
                                 +"{ //\"Connections\" loop "+str(n)) #"Connections" loop
            nrcnt = counters[dm+1]
        else: nrcnt = 0
        s.append((dm+2)*"\t"+"{6}[{1}] += {7}[{3}]*gconns[{4}+{3}*{5}+{1}]".format(\
                                                                                       nextner, nrcnt,\
                                                                                       currner, counters[dm],\
                                                                                       currcon, str(a[n+1]),\
                                                                                       lneurons[n%2], lneurons[(n+1)%2])+";")
        if(a[n+1]>1):
            s.append((dm+1)*"\t"+"}") #"Connections" loop
        s.append((dm+1)*"\t"+"{0}[{1}] = 0.0;".format(lneurons[(n+1)%2], counters[dm]))
        s.append(dm*"\t"+"}") #"Neurons" loop
        currcon += a[n]*a[n+1]
        currner += a[n]
        #dm+=1
    s.append((dm)*"\t"+"result += fabs(gtargets[{0}] - {1}[0]);".format("n_sam", lneurons[n%2])) #"Samples" loop
    s.append((dm)*"\t"+"{0}[0] = 0.0;".format(lneurons[n%2])) #"Samples" loop
    s.append((dm)*"\t"+"gneurons += nvarsd;") #"Samples" loop
    s.append((dm-1)*"\t"+"}") #"Samples" loop
    s.append((dm-1)*"\t"+"results[gid] = result;\n}") #Kernel end
    return "\n".join(s)


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
#s.append("for({0}=0; {0}<{1}; {0}++)".format(counters[dm], str(len(a)))+"{ //\"Layers\" loop")        #Layers loop
    currner = 0
    nextner = 0
    currcon = 0
    n = 0
#nextner += a[n]
#s.append(dm*"\t"+"sums[{1}][{0}]+=xxx;".format(counters[dm], dm-1))
    ss = []
    ses = []
    #shiftsg = [0]+len(a)*[a[0]*samples]
    dcs = [0]+len(a)*[a[0]]
    for n in range(0, len(a)-1):
        s = []
        kname = "runnet"
        s.append("__kernel void "+kname+"(__global float *_gconns, __global float *_"+\
                 lneurons[(n+1)%2]+", __global float *gtargets, __global float *results){")
        s.append(dm*"\t"+"uint gid = get_global_id(0);")
        s.append(dm*"\t"+"float dnr[{0}]".format(a[n])+";// = {"+", ".join(["0.0"])+"};")
        s.append(dm*"\t"+"float lnr[{0}]".format(a[n])+";// = {"+", ".join(["0.0"])+"};")
        s.append(dm*"\t"+"__global float *g{0} = _{0} + gid*DC;".format(lneurons[(n+1)%2])) #Data Count
        s.append(dm*"\t"+"__global float *gconns;")    #Conns shift and Conns number
        s.append(dm*"\t"+"float result = 0.0;")
        #s.append((dm)*"\t"+"__global float gneurons = _gneurons+{0};".format(neuronss[n])) #"Samples" loop
        #dm+=1
        currcon += a[n]*a[n+1]
        nextner += a[n]
        s.append(dm*"\t"+"for(uint {0}=0; {0}<SC; {0}++)".format(counters[dm])+"{ //\"Samples\" loop "+str(n)) #"Samples" loop
        dm+=1
        s.append(dm*"\t"+"gconns = _gconns+CS+CN*gid;")    #Conns shift and Conns number
        s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n]))+"{")
        s.append((dm+1)*"\t"+"{0}[{1}] = {2}{0}[{1}];".format(lneurons[(n+1)%2], counters[dm], ['_', 'g'][int(n>0)]))
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
                se.append((dm-1)*"\t"+"}//\"Samples\" loop "+str(n))
                se.append("}") #"Kernel" end
        currcon += a[n]*a[n+1]
        currner += a[n]
        #dm+=1
        s.append(dm*"\t"+"g{0} += {1};".format(lneurons[(n+1)%2], a[0]))
        s.append((dm)*"\t"+"result += fabs(gtargets[{0}] - {1}[0]);".format(counters[dm-1], lneurons[m%2])) #"Samples" loop
        s.append(dm*"\t"+"for(uint {0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n])))
        s.append((dm+1)*"\t"+"{0}[{1}] = 0.0;".format(lneurons[m%2], counters[dm])) #"Samples" loop
        dm-=1
        s.append((dm)*"\t"+"}//\"Samples\" loop "+str(n))
        s.append((dm)*"\t"+"results[gid] = result;\n}") #Kernel end
        s = ["#define SC {0}".format(samples), "#define DC 0 //Step for prev layer data by each gen".format(dcs[n]), "#define CS {0}".format(sconns[n]), "#define CN {0}".format(conns.sum())] + s
        se = ["#define SC 1", "#define DC {0} //Step for input data".format(a[0]), "#define CS {0}".format(sconns[n]), "#define CN 0"]+se
        ss.append(cmpfunc("\n".join(s)))
        ses.append(cmpfunc("\n".join(se)))
        if n==1:#len(a)-2:
            print("\n".join(s))
            print("\n".join(se))
    return {"ordinal":ss, "finish":ses}

