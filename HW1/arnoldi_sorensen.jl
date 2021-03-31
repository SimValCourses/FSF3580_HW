
"""
    H,V,r=arnoldi_sorensen(A,H,V,r,k,p)

Compute an Arnoldi factorization of length k+p, given a factorization of length k.

This is a reference implementation of Algorithm 3.7 in "Implicit application of polynomial filters in a k-step arnoldi
method", D. C. Sorensen, SIAM J. Matrix Anal. Appl. Vol. 13,
No. 1, pp. 357-385, 1992

The input A, H, V, r, k, p must correspond to an Arnoldi factorization of length k, i.e., satisfy equation (2.1)
"""
function arnoldi_sorensen(A,H,V,r,k,p)

    tol=1e-14;

    for j=1:p
        # Step (1).(1)
        β=norm(r,2);
        if β<tol
            display("Breakdown");
            return;
        end


        # Step (1).(2)   (Expand H and V)
        ekjm1=zeros(size(H,2),1);
        ekjm1[k+j-1]=1;
        H=[H;β*ekjm1'];
        v=r/β;
        V=[V v];

        # Step (1).(3)  (Carry out the matrix vector product)
        w=A*v;

        # Step (1).(4)  (Orthogonalize)
        h=V'*w;
        H=[H h];

        # Step (1).(5)  (Compute new residual)
        r=w-V*h;

        # Step (1).(6)  (Reorthogonalization)
        s=1e60;
        count=0;
        while norm(s)> eps()*norm(r)
            s=V'*r;
            r=r-V*s;
            h=h+s;
            count=count+1;
            if (count > 4)  # Avoid an infinite loop by sanity check
                @warn("Reorthogonalization failed");
            end
        end
    end
    return H,V,r
end
