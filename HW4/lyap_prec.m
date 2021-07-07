function h = lyap_prec(A,n)
    h = @mfun;
    function y = mfun(x)
        X = reshape(x,n,n);
        Y = lyap(A,-X);
        y = Y(:);
    end
end