include("arnoldi_sorensen.jl")

"""
    [V,H,r]=arnupd(A,k,p,tol,v1)

Implicitly restarted Arnoldi method (Sorensen version)

A reference implementation of Algorithm 3.8 in  "Implicit application of polynomial filters in a k-step arnoldi method", D. C. Sorensen, SIAM J. Matrix Anal. Appl. Vol. 13, No. 1, pp. 357-385, 1992


Example:
```julia-repl
n=10;
A=matrixdepot("wathen",n,n);
n=size(A,1);
v1=ones(n);
v1=v1/norm(v1);
k=5;  # Nof. wanted eigenvalues
p=20; # How much should we expand the subspace in every update

include("arnupd.jl");
V,H,r=arnupd(A,k,p,1e-10,v1);
ee= eigvals(H[1:k,1:k]);
ee2=eigvals(Matrix(A));
I2=sortperm(abs.(ee2),rev=true); ee2=ee2[I2]; ee2=ee2[1:k];
I1=sortperm(abs.(ee),rev=true);  ee=ee[I1];
[ee ee2]
```
"""
function arnupd(A,k,p,tol,v1)
    # Step (1) initialize with a trivial Arnoldi factorization
    v1=v1/norm(v1);
    V=v1;
    H=v1'*A*v1;
    r=A*v1-v1*H;

    # Step (2)  (Initiation phase of Arnoldi factorization)
    H,V,r=arnoldi_sorensen(A,H,V,r,1,k-1);

    norm(A*V-V*H-r*[zeros(1,k-1) 1])

    for m=1:100   # Outer loop (i.e. restarts)
        # Step (3).(1)
        print("iteration: ",m,"\n")
        normr=norm(r)
        if (norm(r)<tol)
            restart_count=m-1
            display("finished");
            return V,H,r
        end

        # Step (3).(2) Update the Arnoldi factorization
        H,V,r=arnoldi_sorensen(A,H[1:k,1:k],V[:,1:k],r,k,p);


        # Step (3).(3)  (Select unwanted eigenvalues )
        u=shifts(H,p)

        # Step (3).(4) - Step (3).(7)  Remove unwanted eigenvalues from Arn. fact.
        Q=I;
        for j=1:p
            Qj,Rj=qr(H-u[j]*I);
            H=Qj'*H*Qj;

            Q=Q*Qj;
        end
        VQ=V*Q;
        ekp1=zeros(size(VQ,2)); ekp1[k+1]=1;
        v=VQ*ekp1;
        V=(VQ)*[one(ones(k,k));zeros(p,k)];
        ek=zeros(size(H,2),1); ek[k]=1;
        ekp1=zeros(size(H,1),1); ekp1[k+1]=1;
        β=ekp1'*H*ek;
        ekpp=zeros(size(Q,1),1); ekpp[k+p]=1;
        σ=ekpp'*Q*ek;
        r=v*β+r*σ;

    end
    display("Exceeded nof. outer iterations.");
    return V,H,r
end

function shifts(H,p)
# Return p unwanted eigenvalue approximations of H (in this case the smallest in modulus)
    lv=eigvals(H);
    I=sortperm(abs.(lv));    lv=lv[I];
    # lv is ordered by increasing order of magnitude
    u=lv[1:p];

#    lv=eig(H);
#    [Y,I]=sortrows(-real(lv)); lv=lv(I);
#    % lv is ordered by increasing order of magnitude
#    u=lv(1:p);
    return u
end
