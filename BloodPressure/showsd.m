A = load('lsd');
B = load('osd');
C = abs(A-B);
DIA_Error = mean(C(:,1))
SYS_Error = mean(C(:,2))