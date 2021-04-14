function [Dxx, G_un, F] = assemble_sysmatrices(n)

h = 1/(n+1);
x = h*[1:n];
y = linspace(h,1-h,n);
[XX, YY] = meshgrid(x,y);

G_un = sqrt((XX-0.5).^2 + (YY-0.5).^2);
F = abs(XX-YY);
Dxx = second_der(n,h);
end