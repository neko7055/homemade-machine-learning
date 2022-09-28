data = readmatrix("data.txt");
%data = data(:end,:);
theta = -pi + 2*pi*(0:719)/720;

SIZE = size(data);
for times = 1:SIZE(1)
    
    xy_data = [(data(times,:).* cos(theta))', (data(times,:).* sin(theta))'];
    [Seg,Si_n,S_n] = Segment(xy_data);
    fprintf("frame_idx: %i\n",times)
    cla();
    hold on; 
    plot(xy_data(:,1),xy_data(:,2),'.')
    pause(1)
%{
        for i=1:S_n
            clf;
            hold on; 
            plot(xy_data(:,1),xy_data(:,2),'.')
            plot(xy_data(Seg(1:Si_n(i),i),1),xy_data(Seg(1:Si_n(i),i),2),'bo');
            xlim([-6,8])
            ylim([-4,3])
            %fprintf("%i\n",i)
        end
%}
end