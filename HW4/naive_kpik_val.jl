include("../HW1/arnoldi_val.jl")
function naive_kpik_val(A,b,m)
    Q, H = arnoldi(A,b,m);
    Ainv = inv(Matrix(A))
    G, H = arnoldi(Ainv,b,m);
    U, R = qr(hcat(Q,G));
    U = U[:,findall(abs.(diag(R)) .> 100*eps())]; # remove duplicate b vector
    P = lyap(U'*Matrix(A)*U,U'*(b*b')*U);
    Xt = U*P*U';

    return Xt
end
