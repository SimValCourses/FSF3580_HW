using LinearAlgebra

function hessenberg_red_val(A)
    # Hessenberg reduction carried out
    # following Algorithm 3.2.1
    #
    n=size(A,1)
    for k=1:n-2
        x = A[k+1:n,k];
        ρ = -sign(x[1]);
        α = ρ*norm(x);
        e1 = zeros(n-k,1);
        e1[1] = 1;
        z = x-α.*e1;
        u = z/norm(z);
        PA = A[k+1:n,k:n] - 2*u*(u'*A[k+1:n,k:n]);
        A[1:n,k+1:n] = A[1:n,k+1:n] - 2*(A[1:n,k+1:n]*u)*u';
    end
    return H=A # should be a hessenberg matrix with same eigenvalues as input A
end
