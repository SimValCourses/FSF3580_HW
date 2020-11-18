"""
A simple implementation of the shift-and-invert Arnoldi method.

"""
function shift_inv_arnoldi(A,b,mu,m)

    n=length(b);
    Q=complex(zeros(n,m+1));
    H=complex(zeros(m+1,m));
    Q[:,1]=b/norm(b);

    for k=1:m
        w=(A-mu*I)\Q[:,k]; # Linear system solve
        # Orthogonalize w against columns of Q.
        # Implement this function or replace call with code for orthogonalizatio
        h,β,z=DCGS(Q,w,k);
        #Put Gram-Schmidt coefficients into H
        H[1:(k+1),k]=[h;β];
        # normalize
        Q[:,k+1]=z/β;
    end
    return Q,H
end
