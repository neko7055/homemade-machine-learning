classdef adaboost < handle
    %ADABOOST Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        x
        y
        T
        alpha
        weaklearner
        basic_learner
    end
    
    methods
        function obj = adaboost(x,y,T,basic_learner)
            obj.x = x;
            obj.y = y;
            obj.T = T;
            obj.basic_learner = basic_learner.copy();
        end
        
        function fit(obj,mode)
            if nargin == 1
                mode = 1;
            end
            [N,~]=size(obj.x);
            a = ones(size(obj.y));
            b = ones(size(obj.y));
            a(obj.y~=1) = 0;
            b(obj.y~=-1) = 0;
            a = sum(a);
            b = sum(b);
            D=ones(N,1);
            D(obj.y==1) = 1/(2*a);
            D(obj.y==-1) = 1/(2*b);
            obj.alpha = zeros(obj.T,1);
            obj.weaklearner = cell(1,obj.T);
            for t = 1:obj.T
                fprintf("i-th of weaklearner is traning :%i / %i\n",t,obj.T)
                D = D/sum(D);
                obj.weaklearner{t} = obj.basic_learner.copy(); 
                switch mode
                    case 0
                        obj.weaklearner{t}.fit(obj.x,obj.y,D);
                    case 1
                        idx = randsample(1:N,N,true,D)';
                        obj.weaklearner{t}.fit(obj.x(idx,:),obj.y(idx));
                end
                label = obj.weaklearner{t}.predict(obj.x);
                r = round(D' * (label.*obj.y), 10);
                if (r ==1 && t~=obj.T)
                    obj.alpha(:) = 0;
                    obj.alpha(1) = 1;
                    obj.T = 1;
                    obj.weaklearner{1} = obj.weaklearner{t};
                    fprintf("adaboost training early stopping since strong learner appears\n");
                    return
                end
                obj.alpha(t) = (1/2)*log((1+r)/(1-r));
                D = D.*exp(-obj.alpha(t) * obj.y.*label);
            end
            obj.alpha = obj.alpha./sum(obj.alpha);
        end
        function pred_y = predict(obj,x)
            F = zeros(size(x,1),1);
            for i = 1:obj.T
                F = F + obj.alpha(i)*obj.weaklearner{i}.predict(x);
            end
            pred_y = F > 0;
            pred_y = pred_y*2 -1;
        end
        function pred_y = prob_predict(obj,x)
            F = zeros(size(x,1),1);
            for i = 1:obj.T
                F = F + obj.alpha(i)*obj.weaklearner{i}.predict(x);
            end
            pred_y = 1./(1 + exp(-20*F));
        end
    end
end

