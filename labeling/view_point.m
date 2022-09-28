data = readmatrix("data2.txt");
%data = data(14:end,:);
theta = -pi + 2*pi*(0:719)/720;
first = 1;

fid=fopen('label2.txt','w');%寫入檔案路徑

SIZE = size(data);
for times = 1:SIZE(1)
    

    d = inf;
    d_idx = -1;
    d_target = [];
    d_mid=[];
    xy_data = [(data(times,:).* cos(theta))', (data(times,:).* sin(theta))'];

    [Seg,Si_n,S_n] = Segment(xy_data);
    if times==1
        target_mean = [xy_data(Seg(1:Si_n(first),first),1),xy_data(Seg(1:Si_n(first),first),2)];
        target_mid = median(target_mean,1);
        target_mean = mean(target_mean,1);
        
    end
    clf;
    hold on; 
    plot(xy_data(:,1),xy_data(:,2),'.')   
    for i=1:S_n
            tmp_mean =  [xy_data(Seg(1:Si_n(i),i),1),xy_data(Seg(1:Si_n(i),i),2)];
            tmp_mid = median(tmp_mean,1);
            tmp_mean = mean(tmp_mean,1);
            
            tmp_d = norm(tmp_mean-target_mean);
            if tmp_d < d
                d_idx = i;
                d = tmp_d;
                d_target = tmp_mean;
                d_mid = tmp_mid;
            end
    end
    if d > 0.18
        fprintf("%i\n",-1)
        fprintf(fid,'%i\n',-1);
        plot(xy_data(Seg(1:Si_n(d_idx),d_idx),1),xy_data(Seg(1:Si_n(d_idx),d_idx),2),'bo')
        plot((target_mid(1) +target_mean(1))/2 ,(target_mid(2) +target_mean(2))/2,'ro')
        xlim([-6,8])
        ylim([-4,3])
        pause(0.1)
        continue
    end
    target_mean = d_target;
    target_mid = d_mid;
    fprintf("%i\n",d_idx)
    fprintf(fid,'%i\n',d_idx);
    plot(xy_data(Seg(1:Si_n(d_idx),d_idx),1),xy_data(Seg(1:Si_n(d_idx),d_idx),2),'bo')
    plot((target_mid(1) +target_mean(1))/2 ,(target_mid(2) +target_mean(2))/2,'ro')
    xlim([-6,8])
    ylim([-4,3])
    pause(0.1)
end
fclose(fid);
