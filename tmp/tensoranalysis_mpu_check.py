import sys
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path.cwd()),str(Path.cwd()/'wen_plq_model')]
from tensor_equation_solver import tensor_equation_solver, PAULI, kron_all
from vumpsTransferMatrix.ncon import ncon
I,X,Y,Z=[PAULI[k] for k in 'IXYZ']
m,basis=tensor_equation_solver([[I,I,X.T,X.T],[Z,X,I.T,X.T],[Y,I,Y.T,Z.T],[Y,Y,I.T,I.T]],rng=0)
m=basis[:,0].reshape(2,2,2,2)
m/=m.flat[np.argmax(np.abs(m))]/abs(m.flat[np.argmax(np.abs(m))])
def pair(N,j,p,q):
    ops=[I]*N;ops[j%N]=p;ops[(j+1)%N]=q
    return kron_all(ops)
for N in [2,4,6]:
    U=ncon([m]*N,[[j+1,-j-1,-N-j-1,(j+1)%N+1] for j in range(N)]).reshape(2**N,2**N)
    U/=np.linalg.norm(U)
    F=[pair(N,j,X,Y) for j in range(N)]
    H=[pair(N,j,Y,X) for j in range(N)]
    G=kron_all([X,Y]*(N//2));Gb=kron_all([Y,X]*(N//2));GZ=kron_all([Z]*N)
    print('N',N,'rank',np.linalg.matrix_rank(U), 'unitarity evals',np.round(np.linalg.eigvalsh(U.conj().T@U),5))
    print('G/Gb/Z left-eigen residuals',[(round(np.linalg.norm(Q@U-U),8),round(np.linalg.norm(Q@U+U),8)) for Q in [G,Gb,GZ]])
    print('F commute',[round(np.linalg.norm(Q@U-U@Q),8) for Q in F]);print('H commute',[round(np.linalg.norm(Q@U-U@Q),8) for Q in H])
    for gn,g in [('+',1),('-',-1)]:
        for hn,h in [('+',1),('-',-1)]:
            P=(np.eye(2**N)+g*G)@(np.eye(2**N)+h*Gb)/4
            r=np.linalg.matrix_rank(P)
            print('sector',gn+hn,'U trace',np.trace(P@U)/r,'U commute P',np.linalg.norm(P@U-U@P),'U compressed evals',np.round(np.linalg.eigvals(P@U@P)[np.abs(np.linalg.eigvals(P@U@P))>1e-9],5))
    # Solve signed Pauli intertwiner U F_j = +/- F_k U.
    print('F intertwiners', [[(k,s,round(np.linalg.norm(U@f-s*g@U),8)) for k,g in enumerate(F) for s in [1,-1] if np.linalg.norm(U@f-s*g@U)<1e-8] for f in F])
