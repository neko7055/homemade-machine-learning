function output = transform_data_to_8d(input)
SHAPE = size(input);
if SHAPE(1) == 1
s = 0;
r = 0;
else
[s,r] = circle_fit(input);
end
output = [SHAPE(1) , Std(input) , Width(input) ,s , r,Mean_average_deviation_from_median(input),Linearity(input),Boundary_length(input),Boundary_regularity(input),Mean_angular_difference(input)]; %Mean_average_deviation_from_median(input),Linearity(input),Boundary_length(input),Boundary_regularity(input),Mean_angular_difference(input)

function y = Std(x)
Size = size(x);
if Size(1)==1
y = 0;
return
end
Mean = mean(x,1);
x = x - Mean;
y = norm(x,'fro')/(Size(1)-1)^(1/2);
end

function y = Mean_average_deviation_from_median(x)
Size = size(x);
switch mod(Size(1),2)
    case 0
        Median = (x(Size(1)/2,:) + x(Size(1)/2+1,:))/2;
    case 1
        Median= x((Size(1)+1)/2,:);
end
x = x - Median;
y = mean(vecnorm(x,2,2));
end

function y = Width(x)
    y = norm(x(1,:)-x(end,:));
end

function y = Linearity(x)
if size(x,1)==1
y = 0;
return
end
Mean = mean(x,1);
x = x - Mean;
v = pca(x);    
beta = v(2,1)/v(1,1);
y = sum(   (beta*x(:,1)-x(:,2)).^2   )/(beta^2+1);
end

function [s,r] = circle_fit(x)
    Size = size(x);
    A = [-2*x(:,1) , -2*x(:,2),ones(Size(1),1) ];
    b = [-x(:,1).^2 - x(:,2).^2];
    center = (pinv(A)*b)';
    r = center(end);
    center = center(1:end-1);
    r = (norm(center)^2 - r)^(1/2); 
    s = norm(vecnorm(x-center,2,2) - r)^2;
end
function y = Boundary_length(x)
N = size(x,1);
if N ==1
y = 0;
return
end
y = 0;
for i=1:N-1
y = y + norm(x(i,:) - x(i+1,:));
end
end

function y = Boundary_regularity(x)
N = size(x,1);
y = zeros(N-1,1);
if N ==1
y = 0;
return
end
for i=1:N-1
y(i) = norm(x(i,:) - x(i+1,:));
end
y = std(y);
end

function y = Mean_angular_difference(x)
N = size(x,1);
b = zeros(N-2,1);
if N <3
y = 0;
return
end
for i=1:N-2
    line1 = (x(i,:) - x(i+1,:));
    line2 = (x(i+2,:) - x(i+1,:));
    b(i) = line1* (line2') / (norm(line1)*norm(line2));
    b(i) = acos(b(i));
end
y = mean(b);
end
end