import numpy as np
dm = 0
counters = "ijkzhiuy"
sums = "abcdef"
a = np.array([3, 2, 1], dtype=np.uint32)
s = []
s.append("float sum = 0.0;")
s.append("float sums[{0}];".format(a.sum()))
s.append("for({0}=0; {0}<{1}; {0}++)".format(counters[dm], str(len(a)))+"{")        #Layers loop
currner = 0
nextner = a[0]
currcon = 0
for n in range(0, len(a)-1):
    dm+=1
    s.append(dm*"\t"+"float "+sums[dm]+" = 0.0;")
    #s.append(dm*"\t"+"sums[{1}][{0}]+=xxx;".format(counters[dm], dm-1))
    s.append(dm*"\t"+"for({0}=0; {0}<{1}; {0}++)".format(counters[dm], str(a[n]))+"{") #"Neurons" loop
    s.append((dm+1)*"\t"+"for({0}=0; {0}<{1}; {0}++)".format(counters[dm+1], str(a[n+1]))+"{") #"Connections" loop
    s.append((dm+2)*"\t"+"neurons[{0}+{1}] += neurons[{2}+{3}]*conns[{4}+{3}*{5}+{1}]".format(nextner, counters[dm+1],\
                                                                                   currner, counters[dm],\
                                                                                   currcon, str(a[n+1]))+";\n"+(dm+1)*"\t"+"}") #"Connections" loop
    s.append(dm*"\t"+"float "+sums[dm]+" = 0.0;")
    currcon += a[n]*a[n+1]
    currner = nextner
    nextner += a[n]
print("\n".join(s))
