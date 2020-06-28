function [inG0] = inG(n, f) 
inG0=zeros(n, f);
for m = 1:n
    r=randi(f);
    inG0(m, r)=1;
end
