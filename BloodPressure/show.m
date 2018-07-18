A = load('l1');
B = load('t1');
x = 1:512;
l = 6;
plot(x,A(l,:),'r',x,B(l,:),'g')