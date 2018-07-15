A = load('l1');
B = load('t1');
x = 1:512;
l = 1;
plot(x,A(l,:),'r',x,B(l,:),'g')