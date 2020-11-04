"""
    Q,H=arnoldi(A,b,m)

A simple implementation of the Arnoldi method.
The algorithm will return an Arnoldi "factorization":
Q*H[1:m+1,1:m]-A*Q[:,1:m]=0
where Q is an orthogonal basis of the Krylov subspace
and H a Hessenberg matrix.

The function `my_hw1_gs(Q,w,k)` needs to be available.

Example:
```julia-repl
using Random
A=randn(100,100); b=randn(100);
m=10;
Q,H=arnoldi(A,b,m);
println("1:st should be zero = ", norm(Q*H-A*Q[:,1:m]));
println("2:nd Should be zero = ", norm(Q'*Q-I));
```

"""
function arnoldi(A,b,m,v)

    n=length(b);
    Q=complex(zeros(n,m+1));
    H=complex(zeros(m+1,m));
    Q[:,1]=b/norm(b);

    for k=1:m
        w=A*Q[:,k]; # Matrix-vector product
        h,β,z=GS(Q,w,k,v);
        #Put Gram-Schmidt coefficients into H
        H[1:(k+1),k]=[h;β];
        # normalize
        Q[:,k+1]=z/β;
    end
    return Q,H
end

function arnoldiSI(B,b,m,v)

    n=length(b);
    Q=complex(zeros(n,m+1));
    H=complex(zeros(m+1,m));
    Q[:,1]=b/norm(b);

    for k=1:m
        w=B\Q[:,k]; # Linear system solve
        h,β,z=GS(Q,w,k,v);
        H[1:(k+1),k]=[h;β];
        # normalize
        Q[:,k+1]=z/β;
    end
    return Q,H
end
