classdef bagging < handle
    %ADABOOST Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        x
        y
        T
        learner
        basic_learner
    end
    
    methods
        function obj = bagging(x,y,T,basic_learner)
            obj.x = x;
            obj.y = y;
            obj.T = T;
            obj.basic_learner = basic_learner.copy();
        end
        
        function fit(obj,num_of_sample)
            [N,~]=size(obj.x);
            if nargin == 1
                num_of_sample = N;
            end
            obj.learner = cell(1,obj.T);
            for i = 1:obj.T
                fprintf("i-th of weaklearner is traning :%i / %i\n",i,obj.T)
                idx = randsample(1:N,num_of_sample,true)';
                obj.learner{i} = obj.basic_learner.copy();
                obj.learner{i}.fit(obj.x(idx,:),obj.y(idx));
            end
        end
        function pred_y = predict(obj,x)
            F = zeros(size(x,1),1);
            for i = 1:obj.T
                F = F + obj.learner{i}.predict(x);
            end
            pred_y = F > 0;
            pred_y = pred_y*2 -1;
        end
    end
end

