A = load('l');
B = load('o');
mA = max(A, [], 2);
miA =min(A,[],2);
mB =max(B,[],2);
miB =min(B,[],2);
mC = abs(mA-mB);
miC =abs(miA-miB);
C = abs(A-B);
x = 1:512;
l = 33;
plot(x,A(l,:),'r',x,B(l,:),'g')