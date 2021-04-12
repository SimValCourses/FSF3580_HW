using LinearAlgebra
function errfun(A)
    B = LowerTriangular(A);
    C = Diagonal(A);
    B = B-C;
    err = maximum(maximum(abs.(B)));
    return err
end

function QR_val(A,tol)
    n = size(A,1);
    U = Matrix{Float64}(I, n, n);
    iter = 0;
    global err = 1;
    err_v = [];
    T = deepcopy(A);
    while err > tol
        iter += 1;
        F = qr(T);
        Q = F.Q;
        R = F.R;
        T = R*Q;
        U = U*Q;
        err = errfun(T);
        err_v = [err_v; err]
    end
    return T, U, iter, err_v
end

function shifted_QR_val(A, σ)
    T = deepcopy(A);
    F = qr(T-σ*I);
    Q = F.Q;
    R = F.R;
    T = R*Q + σ*I;
    return T
end
