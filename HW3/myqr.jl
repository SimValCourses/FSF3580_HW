function myqr(A,m)

    n = size(A)[1]
    Am = A;
    errv = zeros(m)

    for i=1:m
        F = qr(Am);
        Am = F.R*F.Q;
        #display(Am)
        errv[i] = maximum(maximum(abs(tril(Am,-1))))
    end

    return Am,F,errv

end

function myqrtol(A,tol)

    n = size(A)[1]
    Am = deepcopy(A);
    err = 1;

    m = 0;
    while err > tol
        m += 1;
        global F = qr(Am);
        Am = F.R*F.Q;
        err = maximum(maximum(abs.(tril(Am,-1))))
    end

    return Am,F,m

    return

end
