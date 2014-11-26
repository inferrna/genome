import numpy as np
def runner(conns, samples, sconns, topology):
    nrs = [np.empty(topology[0], dtype=np.float32), np.empty(topology[0], dtype=np.float32)]
    lconns = np.split(conns, sconns, axis=0)
    result = 0.0
    for i in range(0, len(samples)):
        dnr = nrs[0]
        np.copyto(dnr, samples[i])
        #lcc - layer conns count per neuron, lnc - layer neurons count
        for lc in range(0, len(topology)-1):
            dnr = nrs[lc%2]
            lnr = nrs[(lc+1)%2]
            lnr.fill(0)
            ncc = topology[lc]      #neuron count current
            ncn = topology[lc+1]    #neuron count next
            for n in range(0, ncc):
                for c in range(0, ncn):
                    try:
                        lnr[c] += dnr[n]*lconns[lc][n*ncn+c]
                    except:
                        print("Len(lnr) is", len(lnr), "index is", c)
                        print("Len(dnr) is", len(dnr), "index is", n)
                        print("Len(lconns) is", len(lconns), "index is", lc)
                        if len(lconns)>lc:
                            print("Len(lconns[lc]) is", len(lconns[lc]), "index is", n*ncn+c)
                            print(sconns)
                        exit()

        result += abs(lnr[0]-1.0)
    return result
