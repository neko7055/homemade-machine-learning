clc;clear;
label_data = readmatrix("DataSet\2Circle1.txt");
N = size(label_data,1);
label_data = label_data(randperm(N),:);
label = label_data(:,end);
label(label~=1) = -1;
data = label_data(:,1:end-1);
train_data = data(1:floor(N*0.8),:);
test_data = data(floor(N*0.8) + 1:end,:);
train_label = label(1:floor(N*0.8),:);
test_label = label(floor(N*0.8) + 1:end,:);

classifier =   Bayes_classifier();
classifier.fit(train_data, train_label,15);
%classifier =  DecisionTree(100);
%classifier.fit(train_data, train_label);

%classifier =  adaboost(train_data, train_label,100,RBFNN_QUICK());
%classifier.fit(0);

%classifier =  ExtraTree(train_data, train_label,100,10000);
%classifier.fit();

[X,Y] = meshgrid(-5:.01:5);
X = [X(:),Y(:)];
Z = classifier.predict(X);
[X,Y] = meshgrid(-5:.01:5);
Z = reshape(Z,size(X));
pcolor(X,Y,Z);
shading interp;
hold on
plot(data(label==1,1),data(label==1,2),'k.','MarkerSize',12)
plot(data(label==-1,1),data(label==-1,2),'r.','MarkerSize',12)
axis equal
xlim([-5,5])
ylim([-5,5])
legend('','label = 1','label = -1') 
title('Bayes classifier with gaussian mixture model (use 15 gaussian distributions to approach target distribution)')


pred_y = classifier.predict(train_data);
disp(mean(pred_y ==train_label))

pred_y = classifier.predict(test_data);
disp(mean(pred_y ==test_label))

