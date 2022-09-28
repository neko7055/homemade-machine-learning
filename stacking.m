classdef stacking < handle
    
    properties
        n
        x
        y
        learner
        basic_learner
    end
    
    methods
        function obj = stacking(basic_learner)
            obj.basic_learner = basic_learner.copy();
        end

        function fit(obj,x,y,learner)
            obj.learner = learner;
            obj.n = length(learner);
            
            obj.x = zeros(size(x,1),obj.n);
            for i = 1:obj.n
                try
                obj.x(:,i) = obj.learner{i}.prob_predict(x);
                catch
                obj.x(:,i) = (obj.learner{i}.predict(x)+1)/2;
                end
            end
            obj.y = y;
            obj.basic_learner.fit(obj.x,obj.y);
        end
        function out = copy(obj)
                out = stacking();
                out.x =obj.x;
                out.y =obj.y;
                out.n=obj.n;
                out.learner=obj.learner;
                out.basic_learner = obj.basic_learner;
        end
        function y = predict(obj,x)
            X = zeros(size(x,1),obj.n);
            for i = 1:obj.n
                try
               X(:,i) = obj.learner{i}.prob_predict(x);
                catch
                X(:,i) = (obj.learner{i}.predict(x)+1)/2;
                end
               
            end
            y = obj.basic_learner.predict(X);
        end
    end
end
