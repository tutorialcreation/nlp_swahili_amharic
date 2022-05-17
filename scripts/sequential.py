from math import *
import math
import numpy as np
#REFERENCE
# A Conditional Sequential Test for the Equality of Two Binomial Proportions
# William Q. Meeker, Jr
# Journal of the Royal Statistical Society. Series C (Applied Statistics)
# Vol. 30, No. 2 (1981), pp. 109-115
class ConditionalSPRT:
    def __init__(self,exposed,control,odd_ratio,alpha=0.05,beta=0.10,stop=None):
        """
        - Initialize of the conditionalSprt class
        """
        self.x = exposed
        self.y = control
        self.odd_ratio = odd_ratio
        self.alpha = alpha
        self.beta = beta

    def comb(self,n, k):
        return factorial(n) // factorial(k) // factorial(n - k)

    def lchoose(self,b, j):
        a=[]
        if (type(j) is list) | (isinstance(j,np.ndarray)==True):
            if len(j)<2:
                j=j[0]
        if (type(j) is list) | (isinstance(j,np.ndarray)==True):
            for k in j:
                n=b
                if (0 <= k) & (k<= n):
                    a.append(math.log(self.comb(n,k)))
                else:
                    a.append(0)
        else:
            n=b
            k=j
            if (0 <= k) & (k<= n):
                a.append(math.log(self.comb(n,k)))
            else:
                a.append(0)

        return np.array(a)

    def g(self,x,r,n,t1,t0=1):
        """
        #
        # Meeker's (1981) function `g`, the log probability ratio.
        # 
        """
        return -math.log(self.h(x,r,n,t1))+math.log(self.h(x,r,n,t0))

    def h(self,x,r,n,t=1):
        """
        #
        # Reciprocal of Meeker's (1981) function `h`: the conditional probability of 
        # `x` given `r` and `n`, when the odds ratio is `t`.
        #
        # `x` is his "x1", the number of positives in `n` control trials.
        # `r` is the total number of positives.
        # `n` is the number of (control, treatment) pairs.
        # `t` is the odds ratio.
        #
        """
        return self.f(r,n,t,offset=self.ftermlog(x,r,n,t))

    def f(self,r,n,t,offset=0):
        """#
        # Meeker's (1981) function exp(F(r,n,t)), proportional to the probability of 
        #  `r` (=x1+x2) in `n` paired trials with an odds ratio of `t`.
        #
        # This function does *not* vectorize over its arguments.
        #"""
        upper=max(0,r-n)
        lower=min(n,r)
        rng=list(range(upper,lower+1))
        return np.sum(self.fterm(rng,r,n,t,offset))

    def fterm(self,j,r,n,t,offset=0):
        ftlog=self.ftermlog(j,r,n,t,offset)
        return np.array([math.exp(ex) for ex in ftlog])

    def ftermlog(self,j,r,n,t,offset=0):
        """
        #
        # Up to an additive constant, the log probability that (x1, x1+x2) = (j, r) 
        # in `n` paired trials with odds ratio of `t`.
        #
        # `offset` is used to adjust the result to avoid under/overflow.
        #
        """
        xx=r-j
        lch=self.lchoose(n,j)
        lchdiff=self.lchoose(n,xx)
        lg=np.array(j)*math.log(t)
        lgsum=lch+lchdiff
        lgsum2=lgsum+lg
        lgdiff=lgsum2-offset

        return lgdiff

    def logf(self,r,n,t,offset=0):
        """
        #
        # A protected vesion of log(f), Meeker's function `F`.
        #
        """
        z=self.f(r,n,t,offset)
        if z>0:
            return math.log(z)
        else:
            return np.nan

    def clowerUpper(self,r,n,t1c,t0=1,alpha=0.05,beta=0.10):
        """
        #
        # Meeker's (1981) functions c_L(r,n) and c_U(r,n), the  critical values for x1.
        # 0 <= r <= 2n; t1 >= t0 > 0.
        #
        """
        offset=self.ftermlog(math.ceil(r/2),r,n,t1c)
        z=self.logf(r,n,t1c,self.logf(r,n,t0,offset)+offset)
        a=-math.log(alpha/(1-beta))
        b=math.log(beta/(1-alpha))
        lower=b
        upper=1+a
        return (np.array([lower,upper])+z)/math.log(t1c/t0)
  
    def ConditionalSPRT(self,stop,x,y,t1,n0=None,alpha=0.05,beta=0.10):
        """
        Returns 
        """
        x=np.array(x)
        y=np.array(y)
        l=math.log(beta/(1-alpha))
        u=-math.log(alpha/(1-beta))
        sample_size=min(len(x),len(y))
        n=np.array(range(1,sample_size+1))
        
        if stop!=None:
            n=np.array([z for z in n if z<=stop])
            
        x1=np.cumsum(x[n-1])
        r=x1+np.cumsum(y[n-1])
        stats=np.array(list(map(self.g,x1, r, n, [t1]*len(x1)))) #recurcively calls g
         #
          # Perform the test by finding the first index, if any, at which `stats`
          # falls outside the open interval (l, u).
          #
        clu=list(map(self.clowerUpper,r,n,[t1]*len(r),[1]*len(r),[alpha]*len(r), [beta]*len(r)))
        limits=[]
        for v in clu:
            inArray=[]
            for vin in v:
                inArray.append(math.floor(vin))
            limits.append(np.array(inArray))
        limits=np.array(limits)

        k=np.where((stats>=u) | (stats<=l))
        cvalues=stats[k]
        if cvalues.shape[0]<1:
            k= np.nan
            outcome='Unable to conclude.Needs more sample.'
        else:
            k=np.min(k)
            if stats[k]>=u:
                outcome=f'Exposed group produced a statistically significant increase.'
            else:
                outcome='Their is no statistically significant difference between two test groups'
        if (stop!=None) & (k==np.nan):
          #
          # Truncate at trial stop, using Meeker's H0-conservative formula (2.2).
          # Leave k=NA to indicate the decision was made due to truncation.
          #
            c1=self.clowerUpper(r,stop,t1,alpha,beta)
            c1=math.floor(np.mean(c1)-0.5)
            if x1[n0]<=c1:
                truncate_decision='h0'
                outcome='Maximum Limit Decision. The aproximate decision point shows their is no statistically significant difference between two test groups'
            else:
                truncate_decision='h1'
                outcome=f'Maximum Limit Decision. The aproximate decision point shows exposed group produced a statistically significant increase.'
            truncated=stop
        else:
            truncate_decision='Non'
            truncated=np.nan
        return (outcome,n, k,l,u,truncated,truncate_decision,x1,r,stats,limits)