function [Seg,Si_n,S_n] = Segment(xy)

x = xy(:,1);
y = xy(:,2);

threshold= 0.125;

S_i = 1; % size of segment
S_n = 1; % number of segment



n0ind = find(x~=0 | y~=0);
n_s = size(n0ind);
n_0 = n_s(1);
one_is_end = sqrt( ( x(n0ind(1)) - x(n0ind(end)) )^2 + ( y(n0ind(1)) - y(n0ind(end)) )^2  ) < threshold;

Seg(1,1) = n0ind(1);
for i = 2:n_0
    if sqrt( ( x(n0ind(i)) - x(n0ind(i-1)) )^2 + ( y(n0ind(i)) - y(n0ind(i-1)) )^2  ) < threshold
        S_i = S_i + 1;
        Seg(S_i,S_n) = n0ind(i);
    else
        S_n = S_n + 1;
        S_i = 1;
        Seg(S_i,S_n) = n0ind(i);
    end
end

if one_is_end
    tmp1 = [Seg(Seg(:,1)~=0,1);Seg(Seg(:,end)~=0,end)];
    LEN = max([size(Seg,1),length(tmp1)]);
    tmp = zeros(LEN,1);
    tmp1 = [Seg(Seg(:,1)~=0,1);Seg(Seg(:,end)~=0,end)];
    tmp(1:length(tmp1))=tmp1;
    if size(Seg,1) < LEN
        tmp_seg = Seg;
        Seg = zeros(LEN,S_n);
        for i =1:S_n
            Seg(1:sum(tmp_seg(:,i)~=0),i) = tmp_seg(1:sum(tmp_seg(:,i)~=0),i);
        end
    end
    Seg(:,1) =  tmp;
    Seg(:,end) = [];
    S_n = S_n -1;
end

Si_n = zeros(S_n,1);
for j = 1:S_n
    k = size(find(Seg(:,j)~=0));
    Si_n(j) = k(1);
end

end