classdef Bayes_classifier < handle
    
    properties
        p0
        p1 
        TF
        P_one
        P_zero
        k

    end
    
    methods
        function obj = Bayes_classifier()
        end
        function fit(obj,x,y,k)
            y = (y + 1)/2;
            idx1 = (y==1);
            idx0 = (y==0);
            if nargin == 3
                if isempty(obj.k)
                    obj.k = 5;
                end
            else
                obj.k = k;
            end
            if isempty(obj.TF)
                obj.TF = minimax_tf(x);
            end
            x = obj.TF.tf(x);
            if isempty(idx0)
                obj.p0 = @(x) zeros(size(x,1),1);
                obj.p1 = @(x) ones(size(x,1),1);
                return
            end
            if isempty(idx1)
                obj.p0 = @(x) ones(size(x,1),1);
                obj.p1 = @(x) zeros(size(x,1),1);
                return
            end
            obj.P_zero = mean(idx0);
            obj.P_one = mean(idx1);
            x0 = x(idx0,:);
            x1 = x(idx1,:);
            
            
            obj.p0 = GMM(obj.k,size(x,2));
            obj.p1 = GMM(obj.k,size(x,2));
            obj.p0.fit(x0);
            obj.p1.fit(x1);

        end
        function out = copy(obj)
                out = Bayes_classifier();
                out.p0 =obj.p0;
                out.p1  =obj.p1;
                if ~isempty(obj.TF)
                    out.TF =obj.TF.copy();
                end 
                out.P_one =obj.P_one;
                out.P_zero =obj.P_zero;
                out.k = obj.k;
        end
        function y = predict(obj,x)
            x = obj.TF.tf(x);
            pred_p0 = obj.p0.prod(x).*obj.P_zero;
            pred_p1 = obj.p1.prod(x).*obj.P_one;
            %disp(sum((pred_p0 ==0) .*(pred_p1 ==0) ))
            y = double(pred_p1>pred_p0);
            y = 2*y - 1;
        end
    end
end
