function schur_parlett(A,f)
# The Schur-Parlett method. The function handle f is a scalar
# valued function
    T,Q=schur(Matrix{ComplexF64}(A));   # complex schur form  to make
                                        # it work for complex eigenvalues
    n=size(A,1);
    F=zeros(ComplexF64,n,n)
    for i=1:n
        F[i,i]=f(T[i,i]);
    end
    for p=1:n-1
        #???
    end
    F=Q*F*Q';
    return F
end

A=rand(5,5)
f=z->sin(z)
F=schur_parlett(A,f)
