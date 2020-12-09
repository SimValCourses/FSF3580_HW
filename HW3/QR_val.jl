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
    while err > tol
        iter += 1;
        F = qr(A);
        Q = F.Q;
        R = F.R;
        A = R*Q;
        U = U*Q;
        err = errfun(A);
        err_v = [err_v; err]
    end
    return A, U, iter, err_v
end

function shifted_QR_val(A, σ)
    F = qr(A-σ*I);
    Q = F.Q;
    R = F.R;
    A = R*Q + σ*I;
    return A
end
