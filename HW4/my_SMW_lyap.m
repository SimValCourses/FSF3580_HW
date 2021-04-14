function X = my_SMW_lyap(U1,V1,U2,V2,A,F)
% Solve AX + XA' + M1 X M2' = F,
% where M_i=U_i*V_i' and U_i, V_i are low rank s_i,
% via the Sherman-Morrison-Woodbury matrix-oriented formula
%
% adapted from the script in Hao & Simoncini (2020)

n=size(A,1);
s1=size(U1,2); s2=size(U2,2);

for k=1:s2

end