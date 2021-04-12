Pkg.add("Arpack")
using Arpack

function spectral_abscissa(A);
ev,xv=eigs(A,which=:LR);
I=argmax(real(ev));
return real(ev[I]);
end
