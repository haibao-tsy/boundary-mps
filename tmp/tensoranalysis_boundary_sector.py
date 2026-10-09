exec(open('tmp/tensoranalysis_mpu_check.py').read().split('for N in')[0])
cs=[[X,I,I.T,X.T,X],[I,Y,Y.T,I.T,X],[I,I,X.T,Y.T,Z],[Y,X,I.T,I.T,Z]]
N=4;dim=2**N
def dl(t):
    return ncon([t,t.conj()],[[-1,-3,-5,-7,1],[-2,-4,-6,-8,1]]).reshape(4,4,4,4).transpose(1,0,3,2)
def transfer(a,b):
    return ncon([dl(b),dl(a)]*(N//2),[[j+1,-j-1,(j+1)%N+1,-N-j-1] for j in range(N)]).reshape(4**N,4**N)
def sigma(v):
    return v.reshape([2]*2*N).transpose(list(range(0,2*N,2))+list(range(1,2*N,2))).reshape(dim,dim)
def vector(s):
    return s.reshape([2]*2*N).transpose(np.array([list(range(N)),list(range(N,2*N))]).T.flatten()).reshape(-1)
G=kron_all([X,Y]*(N//2));Gb=kron_all([Y,X]*(N//2))
a,_=tensor_equation_solver(cs,rng=1)
b,_=tensor_equation_solver(cs,rng=2)
a0,_=tensor_equation_solver(cs+[[Y,I,I.T,Y.T,I]],rng=0)
b0,_=tensor_equation_solver(cs+[[X,Y,I.T,I.T,I]],rng=0)
ad,_=tensor_equation_solver(cs+[[I,X,X.T,I.T,I]],rng=0)
for name,a,b in [('genericAB',a,b),('genericAA',a,a),('condensedAB',a0,b0),('currentAB',ad,b0)]:
    T=transfer(a,b)@transfer(b,a)
    print('case',name)
    for gv in [1,-1]:
        for hv in [1,-1]:
            P=(np.eye(dim)+gv*G)@(np.eye(dim)+hv*Gb)/4
            v=vector(P);v/=np.linalg.norm(v)
            for step in range(500):
                v1=T@v
                v1=vector(P@sigma(v1)@P)
                lam=np.linalg.norm(v1)
                if lam<1e-25: break
                v=v1/lam
            s=sigma(v);s/=np.trace(s)
            print('sector',gv,hv,'lam',lam,'eigres',np.linalg.norm(T@v-lam*v),'onesided',np.linalg.norm(G@s-gv*s),np.linalg.norm(Gb@s-hv*s))
    v=vector(np.eye(dim));v/=np.linalg.norm(v)
    for step in range(500):
        v=T@v;v/=np.linalg.norm(v)
    s=sigma(v);s/=np.trace(s)
    print('full fixed sector weights',[np.trace(s@(np.eye(dim)+gv*G)@(np.eye(dim)+hv*Gb)/4) for gv in [1,-1] for hv in [1,-1]])
